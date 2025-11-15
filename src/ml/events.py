"""
Event data handling for TSLA earnings, deliveries, and macro events
Integrates with APIs for real-time event calendars
"""

import pandas as pd
import numpy as np
import torch
from typing import Dict, List, Any
from pathlib import Path
from datetime import datetime, timedelta
import requests
import sys

# Add parent directory to path
parent_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(parent_dir))

import config
from .base import EventHandler


class TSLAEventsHandler(EventHandler):
    """
    Handler for TSLA-specific events (earnings, deliveries)
    Loads from user-provided CSV with beat/miss data
    """

    def __init__(self, events_file: str = None):
        self.events_file = events_file or config.TSLA_EVENTS_FILE
        self.events_df = None
        self.event_types = {
            'earnings': 0,
            'delivery': 1,
            'production': 2,
            'other': 3
        }

    def load_events(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Load TSLA events from CSV
        Expected format: date, event_type, expected, actual, beat_miss
        """
        if not Path(self.events_file).exists():
            print(f"Warning: Events file not found: {self.events_file}")
            print("Creating empty events DataFrame. Please provide tsla_events.csv")
            return pd.DataFrame(columns=['date', 'event_type', 'expected', 'actual', 'beat_miss'])

        df = pd.read_csv(self.events_file)
        df['date'] = pd.to_datetime(df['date'])

        # Check CSV coverage (warn if outdated)
        self._check_csv_coverage(df)

        # Filter by date range
        if start_date:
            df = df[df['date'] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df['date'] <= pd.to_datetime(end_date)]

        self.events_df = df
        return df

    def _check_csv_coverage(self, df: pd.DataFrame):
        """Warn if CSV coverage is ending soon (within 90 days)"""
        if df.empty:
            return

        from datetime import datetime
        max_event_date = df['date'].max()
        current_date = datetime.now()

        days_remaining = (max_event_date - current_date).days

        if days_remaining < 90:  # Less than 3 months coverage
            print(f"\n{'='*70}")
            print(f"  ⚠️  EVENT DATA COVERAGE WARNING")
            print(f"{'='*70}")
            print(f"  Event CSV coverage ends: {max_event_date.strftime('%Y-%m-%d')}")
            print(f"  Days remaining: {days_remaining} days")
            print(f"\n  Action Required:")
            print(f"  1. Update {Path(self.events_file).name} with {max_event_date.year + 1} events")
            print(f"  2. Add upcoming TSLA earnings dates (check investor relations)")
            print(f"  3. Add {max_event_date.year + 1} FOMC schedule (check federalreserve.gov)")
            print(f"  4. See SPEC.md 'Event Data Maintenance' section for instructions")
            print(f"{'='*70}\n")
        elif days_remaining < 180:  # 3-6 months
            print(f"ℹ️  Event CSV coverage until {max_event_date.strftime('%Y-%m-%d')} ({days_remaining} days remaining)")

    def get_events_for_date(self, date: str, lookback_days: int = 7) -> List[Dict]:
        """
        Get events within lookback window of a specific date
        Returns list of event dictionaries
        """
        if self.events_df is None:
            self.load_events()

        target_date = pd.to_datetime(date)
        start_date = target_date - timedelta(days=lookback_days)
        end_date = target_date + timedelta(days=config.EVENT_LOOKAHEAD_DAYS)

        # Filter events in window
        mask = (self.events_df['date'] >= start_date) & (self.events_df['date'] <= end_date)
        events = self.events_df[mask].to_dict('records')

        # Calculate days until/since event
        for event in events:
            days_diff = (event['date'] - target_date).days
            event['days_until_event'] = days_diff

        return events

    def embed_events(self, events: List[Dict]) -> torch.Tensor:
        """
        Convert events to tensor embeddings
        Format: [event_type_one_hot, days_until, beat_miss_encoding, surprise_magnitude]
        """
        if not events:
            # Return zero vector if no events
            return torch.zeros(1, 10)  # 4 (event types) + 1 (days) + 1 (beat/miss) + 1 (magnitude) + 3 (padding)

        embeddings = []
        for event in events:
            # One-hot encode event type
            event_type_idx = self.event_types.get(event['event_type'], 3)
            event_type_onehot = [0.0] * 4
            event_type_onehot[event_type_idx] = 1.0

            # Normalize days until event (-7 to +7 → -1 to +1)
            days_normalized = event['days_until_event'] / 7.0

            # Beat/miss encoding (-1 for miss, 0 for meet, 1 for beat)
            beat_miss_map = {'miss': -1.0, 'meet': 0.0, 'beat': 1.0}
            beat_miss = beat_miss_map.get(event.get('beat_miss', 'meet'), 0.0)

            # Surprise magnitude (actual - expected) / expected
            expected = event.get('expected', 0)
            actual = event.get('actual', expected)
            if expected != 0:
                surprise = (actual - expected) / abs(expected)
            else:
                surprise = 0.0

            # Combine into embedding vector
            embedding = event_type_onehot + [days_normalized, beat_miss, surprise]
            embeddings.append(embedding)

        # Average embeddings if multiple events
        embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)
        return embeddings_tensor.mean(dim=0, keepdim=True)


class MacroEventsHandler(EventHandler):
    """
    Handler for macro economic events (FOMC, CPI, NFP, etc.)
    Pulls from economic calendar APIs
    """

    def __init__(self, api_key: str = None):
        self.api_key = api_key or config.MACRO_EVENTS_API_KEY
        self.events_df = None
        self.event_types = {
            'fomc': 0,
            'cpi': 1,
            'nfp': 2,  # Non-farm payrolls
            'gdp': 3,
            'unemployment': 4,
            'retail_sales': 5,
            'housing': 6,
            'pmi': 7,
            'fed_speech': 8,
            'other': 9
        }

        # Hardcoded events list (can be updated via API)
        self._load_hardcoded_events()

    def _load_hardcoded_events(self):
        """
        Load comprehensive list of known macro events
        This is a fallback if API is not configured
        """
        # Sample macro events (would be extended with full calendar)
        events_data = [
            # FOMC Meetings (2015-2025)
            {'date': '2023-01-31', 'event_type': 'fomc', 'description': 'FOMC Rate Decision'},
            {'date': '2023-03-22', 'event_type': 'fomc', 'description': 'FOMC Rate Decision'},
            {'date': '2023-05-03', 'event_type': 'fomc', 'description': 'FOMC Rate Decision'},
            {'date': '2023-06-14', 'event_type': 'fomc', 'description': 'FOMC Rate Decision'},
            {'date': '2023-07-26', 'event_type': 'fomc', 'description': 'FOMC Rate Decision'},
            {'date': '2023-09-20', 'event_type': 'fomc', 'description': 'FOMC Rate Decision'},
            {'date': '2023-11-01', 'event_type': 'fomc', 'description': 'FOMC Rate Decision'},
            {'date': '2023-12-13', 'event_type': 'fomc', 'description': 'FOMC Rate Decision'},
            {'date': '2024-01-31', 'event_type': 'fomc', 'description': 'FOMC Rate Decision'},
            {'date': '2024-03-20', 'event_type': 'fomc', 'description': 'FOMC Rate Decision'},
            {'date': '2024-05-01', 'event_type': 'fomc', 'description': 'FOMC Rate Decision'},
            {'date': '2024-06-12', 'event_type': 'fomc', 'description': 'FOMC Rate Decision'},
            {'date': '2024-07-31', 'event_type': 'fomc', 'description': 'FOMC Rate Decision'},
            {'date': '2024-09-18', 'event_type': 'fomc', 'description': 'FOMC Rate Decision'},
            {'date': '2024-11-07', 'event_type': 'fomc', 'description': 'FOMC Rate Decision'},
            {'date': '2024-12-18', 'event_type': 'fomc', 'description': 'FOMC Rate Decision'},
            # CPI Reports (monthly, first weeks)
            # NFP (Non-Farm Payrolls - first Friday of each month)
            # ... (would be extended with complete calendar)
        ]

        self.events_df = pd.DataFrame(events_data)
        self.events_df['date'] = pd.to_datetime(self.events_df['date'])

    def load_events(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Load macro events from API or hardcoded list"""
        if self.api_key:
            # TODO: Implement API fetch (FRED, Alpha Vantage, etc.)
            print("API integration not yet implemented, using hardcoded events")

        # Filter by date
        df = self.events_df.copy()
        if start_date:
            df = df[df['date'] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df['date'] <= pd.to_datetime(end_date)]

        return df

    def get_events_for_date(self, date: str, lookback_days: int = 7) -> List[Dict]:
        """Get macro events within window of date"""
        if self.events_df is None:
            self._load_hardcoded_events()

        target_date = pd.to_datetime(date)
        start_date = target_date - timedelta(days=lookback_days)
        end_date = target_date + timedelta(days=config.EVENT_LOOKAHEAD_DAYS)

        mask = (self.events_df['date'] >= start_date) & (self.events_df['date'] <= end_date)
        events = self.events_df[mask].to_dict('records')

        for event in events:
            days_diff = (event['date'] - target_date).days
            event['days_until_event'] = days_diff

        return events

    def embed_events(self, events: List[Dict]) -> torch.Tensor:
        """
        Convert macro events to tensor embeddings
        Format: [event_type_one_hot (10), days_until]
        """
        if not events:
            return torch.zeros(1, 11)  # 10 event types + 1 days

        embeddings = []
        for event in events:
            # One-hot encode event type
            event_type_idx = self.event_types.get(event['event_type'], 9)
            event_type_onehot = [0.0] * 10
            event_type_onehot[event_type_idx] = 1.0

            # Normalize days
            days_normalized = event['days_until_event'] / 7.0

            embedding = event_type_onehot + [days_normalized]
            embeddings.append(embedding)

        embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)
        return embeddings_tensor.mean(dim=0, keepdim=True)


class CombinedEventsHandler(EventHandler):
    """
    Combines TSLA and macro events into unified handler
    """

    def __init__(self, tsla_file: str = None, macro_api_key: str = None):
        self.tsla_handler = TSLAEventsHandler(tsla_file)
        self.macro_handler = MacroEventsHandler(macro_api_key)

    def load_events(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Load both TSLA and macro events"""
        tsla_events = self.tsla_handler.load_events(start_date, end_date)
        macro_events = self.macro_handler.load_events(start_date, end_date)

        tsla_events['source'] = 'tsla'
        macro_events['source'] = 'macro'

        combined = pd.concat([tsla_events, macro_events], ignore_index=True)
        combined = combined.sort_values('date')

        return combined

    def get_events_for_date(self, date: str, lookback_days: int = 7) -> List[Dict]:
        """Get all events for a date"""
        tsla_events = self.tsla_handler.get_events_for_date(date, lookback_days)
        macro_events = self.macro_handler.get_events_for_date(date, lookback_days)

        all_events = tsla_events + macro_events
        all_events.sort(key=lambda x: abs(x['days_until_event']))

        return all_events

    def embed_events(self, events: List[Dict]) -> torch.Tensor:
        """
        Embed all events as concatenated vector
        Returns: [tsla_embedding (7), macro_embedding (11)] = 18 dims
        """
        # Separate by source
        tsla_events = [e for e in events if e.get('source') == 'tsla' or 'beat_miss' in e]
        macro_events = [e for e in events if e.get('source') == 'macro' or 'beat_miss' not in e]

        tsla_embed = self.tsla_handler.embed_events(tsla_events)
        macro_embed = self.macro_handler.embed_events(macro_events)

        # Pad to consistent dimensions
        if tsla_embed.shape[1] < 10:
            padding = torch.zeros(1, 10 - tsla_embed.shape[1])
            tsla_embed = torch.cat([tsla_embed, padding], dim=1)

        combined_embed = torch.cat([tsla_embed, macro_embed], dim=1)
        return combined_embed
