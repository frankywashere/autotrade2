"""
v5.2 Event System for Channel Duration Prediction

Fetches and embeds upcoming market events (FOMC, earnings, deliveries)
for integration with the hierarchical model.

Analogy: Train schedule showing known disruptions ahead.
"""

from datetime import datetime, date, timedelta
from typing import List, Dict, Optional
import torch
import torch.nn as nn
import numpy as np

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

# Import FOMC scraper
try:
    from .fomc_calendar import get_next_fomc_meeting, get_all_fomc_dates_2025
    HAS_FOMC = True
except ImportError:
    HAS_FOMC = False


class LiveEventFetcher:
    """
    Fetch upcoming market events for live inference.

    Analogy: Train schedule showing known disruptions ahead.
    """

    # Event type IDs (for embedding lookup)
    EVENT_TYPES = {
        'fomc': 0,
        'earnings': 1,
        'delivery': 2,
        'cpi': 3,
        'nfp': 4,  # Non-farm payrolls
        'other': 5,
    }

    def __init__(self, finnhub_api_key: Optional[str] = None):
        """
        Initialize event fetcher.

        Args:
            finnhub_api_key: Optional Finnhub API key for earnings data
        """
        self.finnhub_key = finnhub_api_key

        # FOMC dates from scraper
        if HAS_FOMC:
            try:
                self.fomc_dates = get_all_fomc_dates_2025()
            except Exception:
                self.fomc_dates = []
        else:
            self.fomc_dates = []

        # TSLA delivery dates (predictable quarterly: early Jan/Apr/Jul/Oct)
        # Update these annually
        self.delivery_dates = [
            date(2025, 1, 2), date(2025, 4, 2),
            date(2025, 7, 2), date(2025, 10, 2),
            date(2026, 1, 2), date(2026, 4, 2),
        ]

        # CPI release dates (usually mid-month)
        # These are approximate - actual dates vary
        self.cpi_dates = [
            date(2025, 1, 15), date(2025, 2, 12), date(2025, 3, 12),
            date(2025, 4, 10), date(2025, 5, 13), date(2025, 6, 11),
            date(2025, 7, 11), date(2025, 8, 13), date(2025, 9, 11),
            date(2025, 10, 10), date(2025, 11, 13), date(2025, 12, 11),
        ]

    def fetch_upcoming_events(
        self,
        ticker: str = 'TSLA',
        days_ahead: int = 30,
        as_of_date: date = None
    ) -> List[Dict]:
        """
        Fetch all upcoming events within days_ahead.

        Args:
            ticker: Stock ticker (default 'TSLA')
            days_ahead: How many days to look ahead
            as_of_date: Reference date (default: today)

        Returns:
            List of event dicts with keys:
                - type: Event type string
                - type_id: Integer for embedding lookup
                - date: ISO date string
                - days_until: Days from reference date
        """
        events = []
        today = as_of_date or date.today()

        # 1. FOMC meetings (from scraper)
        if HAS_FOMC:
            try:
                next_fomc = get_next_fomc_meeting()
                if 'days_until' in next_fomc and next_fomc['days_until'] <= days_ahead:
                    events.append({
                        'type': 'fomc',
                        'type_id': self.EVENT_TYPES['fomc'],
                        'date': next_fomc['next_meeting'],
                        'days_until': next_fomc['days_until'],
                    })
            except Exception:
                pass

        # 2. TSLA earnings (Finnhub API if available)
        if self.finnhub_key and HAS_REQUESTS:
            earnings = self._fetch_earnings(ticker, days_ahead, today)
            if earnings:
                events.append(earnings)

        # 3. TSLA deliveries (predictable quarterly)
        for delivery_date in self.delivery_dates:
            days = (delivery_date - today).days
            if 0 <= days <= days_ahead:
                events.append({
                    'type': 'delivery',
                    'type_id': self.EVENT_TYPES['delivery'],
                    'date': delivery_date.isoformat(),
                    'days_until': days,
                })
                break  # Only next delivery

        # 4. CPI releases
        for cpi_date in self.cpi_dates:
            days = (cpi_date - today).days
            if 0 <= days <= days_ahead:
                events.append({
                    'type': 'cpi',
                    'type_id': self.EVENT_TYPES['cpi'],
                    'date': cpi_date.isoformat(),
                    'days_until': days,
                })
                break  # Only next CPI

        # Sort by proximity
        events.sort(key=lambda x: x['days_until'])

        return events

    def _fetch_earnings(
        self,
        ticker: str,
        days_ahead: int,
        as_of_date: date
    ) -> Optional[Dict]:
        """Fetch next earnings date from Finnhub."""
        if not HAS_REQUESTS:
            return None

        try:
            url = "https://finnhub.io/api/v1/calendar/earnings"
            params = {
                'symbol': ticker,
                'from': as_of_date.isoformat(),
                'to': (as_of_date + timedelta(days=days_ahead)).isoformat(),
                'token': self.finnhub_key,
            }

            response = requests.get(url, params=params, timeout=10)
            data = response.json()

            if 'earningsCalendar' in data and len(data['earningsCalendar']) > 0:
                next_earnings = data['earningsCalendar'][0]['date']
                earnings_date = datetime.strptime(next_earnings, '%Y-%m-%d').date()
                days = (earnings_date - as_of_date).days

                return {
                    'type': 'earnings',
                    'type_id': self.EVENT_TYPES['earnings'],
                    'date': next_earnings,
                    'days_until': days,
                }
        except Exception as e:
            print(f"Warning: Finnhub earnings fetch failed: {e}")

        return None

    def get_events_for_training(
        self,
        timestamp: datetime,
        days_ahead: int = 30
    ) -> List[Dict]:
        """
        Get events relative to a historical timestamp (for training).

        Uses stored event dates rather than live fetching.

        Args:
            timestamp: Historical datetime
            days_ahead: How many days to look ahead

        Returns:
            List of event dicts
        """
        ref_date = timestamp.date() if hasattr(timestamp, 'date') else timestamp
        events = []

        # FOMC dates
        for fomc_date in self.fomc_dates:
            if isinstance(fomc_date, str):
                fomc_date = datetime.strptime(fomc_date, '%Y-%m-%d').date()
            days = (fomc_date - ref_date).days
            if 0 <= days <= days_ahead:
                events.append({
                    'type': 'fomc',
                    'type_id': self.EVENT_TYPES['fomc'],
                    'date': fomc_date.isoformat(),
                    'days_until': days,
                })

        # Delivery dates
        for delivery_date in self.delivery_dates:
            days = (delivery_date - ref_date).days
            if 0 <= days <= days_ahead:
                events.append({
                    'type': 'delivery',
                    'type_id': self.EVENT_TYPES['delivery'],
                    'date': delivery_date.isoformat(),
                    'days_until': days,
                })

        # CPI dates
        for cpi_date in self.cpi_dates:
            days = (cpi_date - ref_date).days
            if 0 <= days <= days_ahead:
                events.append({
                    'type': 'cpi',
                    'type_id': self.EVENT_TYPES['cpi'],
                    'date': cpi_date.isoformat(),
                    'days_until': days,
                })

        events.sort(key=lambda x: x['days_until'])
        return events[:3]  # Top 3 nearest events


class EventEmbedding(nn.Module):
    """
    Convert upcoming events to learned embedding.

    Analogy: The looming shadow - events cast longer shadows as they approach.
    """

    def __init__(self, event_types: int = 6, embed_dim: int = 32):
        """
        Initialize event embedding network.

        Args:
            event_types: Number of distinct event types
            embed_dim: Output embedding dimension
        """
        super().__init__()

        self.embed_dim = embed_dim

        # Event type embedding
        self.type_embed = nn.Embedding(event_types, 16)

        # Timing features encoder
        # Input: [days_until_normalized, urgency, decay]
        self.timing_net = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
        )

        # Fusion: combine type + timing
        self.fusion = nn.Sequential(
            nn.Linear(32, embed_dim),
            nn.ReLU(),
        )

        # No-event embedding (learnable)
        self.no_event_embed = nn.Parameter(torch.zeros(embed_dim))

    def forward(
        self,
        events: List[Dict],
        batch_size: int,
        device: torch.device
    ) -> torch.Tensor:
        """
        Encode events into embedding.

        Args:
            events: List of event dicts from LiveEventFetcher
            batch_size: Current batch size
            device: Target device

        Returns:
            [batch_size, embed_dim] embedding
        """
        if not events or len(events) == 0:
            # No events: return learned no-event embedding
            return self.no_event_embed.unsqueeze(0).expand(batch_size, -1)

        # Encode top 3 nearest events
        top_events = sorted(events, key=lambda x: x['days_until'])[:3]

        embeddings = []
        for event in top_events:
            # Type embedding
            type_id = torch.tensor([event['type_id']], device=device)
            type_emb = self.type_embed(type_id)  # [1, 16]

            # Timing features
            days = event['days_until']
            timing = torch.tensor([[
                days / 30.0,                        # Normalized days (0-1 for 30 days)
                1.0 / (abs(days) + 1),              # Urgency (closer = higher)
                float(np.exp(-abs(days) / 7.0)),   # Decay (7-day half-life)
            ]], device=device, dtype=torch.float32)
            timing_emb = self.timing_net(timing)  # [1, 16]

            # Combine type + timing
            event_emb = torch.cat([type_emb, timing_emb], dim=-1)  # [1, 32]
            embeddings.append(self.fusion(event_emb))  # [1, embed_dim]

        # Average all event embeddings
        combined = torch.stack(embeddings).mean(dim=0)  # [1, embed_dim]

        # Expand to batch size
        return combined.expand(batch_size, -1)

    def forward_batch(
        self,
        events_batch: List[List[Dict]],
        device: torch.device
    ) -> torch.Tensor:
        """
        Process a batch of event lists (for training).

        Args:
            events_batch: List of event lists, one per sample in batch
            device: Target device

        Returns:
            [batch_size, embed_dim] embeddings
        """
        batch_size = len(events_batch)
        embeddings = []

        for events in events_batch:
            # Get embedding for single sample
            emb = self.forward(events, batch_size=1, device=device)
            embeddings.append(emb)

        return torch.cat(embeddings, dim=0)  # [batch_size, embed_dim]


class VIXSequenceLoader:
    """
    Load and prepare VIX sequences for the VIX CfC layer.

    Provides daily VIX data with derived features for temporal processing.
    """

    # VIX derived features (total 11 features)
    # OHLC (4) + derived (7)
    VIX_FEATURES = [
        'vix_open', 'vix_high', 'vix_low', 'vix_close',  # OHLC
        'vix_rsi_14',           # 14-day RSI
        'vix_percentile_60d',   # 60-day percentile
        'vix_percentile_252d',  # 252-day (1 year) percentile
        'vix_change_1d',        # 1-day change
        'vix_change_5d',        # 5-day change
        'vix_spike_flag',       # >15% daily move
        'vix_regime',           # 0=low, 1=medium, 2=high, 3=extreme
    ]

    VIX_INPUT_SIZE = len(VIX_FEATURES)  # 11

    def __init__(self, vix_csv_path: str = None):
        """
        Initialize VIX loader.

        Args:
            vix_csv_path: Path to VIX_History.csv
        """
        self.vix_csv_path = vix_csv_path
        self.vix_data = None

        if vix_csv_path:
            self._load_vix_data()

    def _load_vix_data(self):
        """Load and preprocess VIX data from CSV."""
        import pandas as pd

        try:
            # Try different date column names (Date vs DATE)
            df = pd.read_csv(self.vix_csv_path)

            # Find date column (case-insensitive)
            date_col = None
            for col in df.columns:
                if col.lower() == 'date':
                    date_col = col
                    break

            if date_col:
                df[date_col] = pd.to_datetime(df[date_col])
                df.set_index(date_col, inplace=True)
                df.sort_index(inplace=True)
            else:
                raise ValueError("No Date column found in VIX CSV")

            # Rename columns to standard format (handle both cases)
            df = df.rename(columns={
                'OPEN': 'vix_open',
                'HIGH': 'vix_high',
                'LOW': 'vix_low',
                'CLOSE': 'vix_close',
                # Also handle lowercase
                'Open': 'vix_open',
                'High': 'vix_high',
                'Low': 'vix_low',
                'Close': 'vix_close',
            })

            # Compute derived features
            df['vix_rsi_14'] = self._compute_rsi(df['vix_close'], 14)
            df['vix_percentile_60d'] = df['vix_close'].rolling(60).apply(
                lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-8)
            )
            df['vix_percentile_252d'] = df['vix_close'].rolling(252).apply(
                lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-8)
            )
            df['vix_change_1d'] = df['vix_close'].pct_change(1)
            df['vix_change_5d'] = df['vix_close'].pct_change(5)
            df['vix_spike_flag'] = (df['vix_change_1d'].abs() > 0.15).astype(float)

            # Regime classification
            df['vix_regime'] = 0  # Default: low
            df.loc[df['vix_close'] > 20, 'vix_regime'] = 1  # Medium
            df.loc[df['vix_close'] > 30, 'vix_regime'] = 2  # High
            df.loc[df['vix_close'] > 40, 'vix_regime'] = 3  # Extreme

            # Fill NaN with forward fill then zeros
            df = df.fillna(method='ffill').fillna(0)

            self.vix_data = df[self.VIX_FEATURES]

        except Exception as e:
            print(f"Warning: Failed to load VIX data: {e}")
            self.vix_data = None

    def _compute_rsi(self, series, period=14):
        """Compute RSI indicator."""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-8)
        return 100 - (100 / (1 + rs))

    def get_sequence(
        self,
        as_of_date: date,
        sequence_length: int = 90
    ) -> np.ndarray:
        """
        Get VIX sequence ending at as_of_date.

        Args:
            as_of_date: End date for sequence
            sequence_length: Number of days to include

        Returns:
            [sequence_length, 11] numpy array
        """
        if self.vix_data is None:
            # Return zeros if no data available
            return np.zeros((sequence_length, self.VIX_INPUT_SIZE))

        # Find the date in our data
        if isinstance(as_of_date, datetime):
            as_of_date = as_of_date.date()

        try:
            # Get index position
            idx = self.vix_data.index.get_indexer([as_of_date], method='pad')[0]

            if idx < 0:
                return np.zeros((sequence_length, self.VIX_INPUT_SIZE))

            # Get sequence
            start_idx = max(0, idx - sequence_length + 1)
            seq = self.vix_data.iloc[start_idx:idx + 1].values

            # Pad if needed
            if len(seq) < sequence_length:
                padding = np.zeros((sequence_length - len(seq), self.VIX_INPUT_SIZE))
                seq = np.vstack([padding, seq])

            return seq.astype(np.float32)

        except Exception as e:
            print(f"Warning: Failed to get VIX sequence: {e}")
            return np.zeros((sequence_length, self.VIX_INPUT_SIZE))

    def get_live_vix(self) -> Dict:
        """
        Fetch current VIX value (for live inference).

        Returns:
            Dict with VIX OHLC and derived features
        """
        try:
            import yfinance as yf

            vix = yf.download("^VIX", period="2d", interval="1d", progress=False)
            if len(vix) > 0:
                latest = vix.iloc[-1]
                return {
                    'vix_open': float(latest['Open']),
                    'vix_high': float(latest['High']),
                    'vix_low': float(latest['Low']),
                    'vix_close': float(latest['Close']),
                    'vix_change_1d': float((latest['Close'] - vix.iloc[-2]['Close']) / vix.iloc[-2]['Close']) if len(vix) > 1 else 0,
                }
        except Exception as e:
            print(f"Warning: Failed to fetch live VIX: {e}")

        return None
