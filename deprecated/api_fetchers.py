"""
API Fetchers for Event Data

Fetches TSLA earnings, FOMC meetings, and macro economic events from various APIs.

APIs Used:
- Alpha Vantage: TSLA earnings calendar (25 calls/day free tier)
- FRED: Macro economic data (unlimited with free API key)
- Federal Reserve website: FOMC schedule (scraping)
- Calculated: CPI, NFP, Quad Witching (predictable schedules)
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import json
import time
from typing import List, Dict, Optional
import sys

# Add parent to path
parent_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(parent_dir))

import config


class APICache:
    """Simple file-based cache for API responses"""

    def __init__(self, cache_file: str = "data/.event_api_cache.json"):
        self.cache_file = Path(cache_file)
        self.cache_file.parent.mkdir(exist_ok=True)
        self.cache = self._load()

    def _load(self) -> dict:
        """Load cache from file"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def _save(self):
        """Save cache to file"""
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f, indent=2)

    def get(self, key: str, max_age_days: int = 30) -> Optional[dict]:
        """Get cached data if not expired"""
        if key not in self.cache:
            return None

        cached_data = self.cache[key]
        cache_date = datetime.fromisoformat(cached_data['timestamp'])
        age_days = (datetime.now() - cache_date).days

        if age_days > max_age_days:
            return None  # Expired

        return cached_data['data']

    def set(self, key: str, data: dict):
        """Cache data with timestamp"""
        self.cache[key] = {
            'timestamp': datetime.now().isoformat(),
            'data': data
        }
        self._save()


class AlphaVantageClient:
    """
    Alpha Vantage API client for TSLA earnings data.

    Free tier: 25 calls/day, 5 calls/minute
    """

    def __init__(self, api_key: str = None):
        self.api_key = api_key or config.ALPHA_VANTAGE_API_KEY
        self.base_url = "https://www.alphavantage.co/query"
        self.cache = APICache()

    def fetch_earnings_calendar(self, symbol: str = "TSLA", horizon: str = "12month") -> pd.DataFrame:
        """
        Fetch earnings calendar for TSLA.

        Args:
            symbol: Stock symbol (default: TSLA)
            horizon: Lookback/lookahead period (3month, 6month, 12month)

        Returns:
            DataFrame with columns: date, event_type, expected, actual, beat_miss, category
        """
        if not self.api_key:
            print("⚠️  Alpha Vantage API key not configured. Skipping earnings fetch.")
            return pd.DataFrame()

        # Check cache (30 day expiry)
        cache_key = f"alpha_vantage_earnings_{symbol}_{horizon}"
        cached = self.cache.get(cache_key, max_age_days=30)
        if cached:
            print(f"   ✓ Using cached Alpha Vantage earnings data (age: <30 days)")
            return pd.DataFrame(cached)

        print(f"   📡 Fetching TSLA earnings from Alpha Vantage...")

        params = {
            'function': 'EARNINGS_CALENDAR',
            'symbol': symbol,
            'horizon': horizon,
            'apikey': self.api_key
        }

        try:
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()

            # Alpha Vantage returns CSV format
            from io import StringIO
            df = pd.read_csv(StringIO(response.text))

            if df.empty:
                print("   ⚠️  No earnings data returned from Alpha Vantage")
                return pd.DataFrame()

            # Transform to our format
            events = []
            for _, row in df.iterrows():
                date = pd.to_datetime(row['reportDate'])

                # Determine beat/miss if actual exists
                beat_miss = 'neutral'
                expected_eps = row.get('estimate', 0.0)
                actual_eps = row.get('reportedEPS', 0.0)

                if pd.notna(actual_eps) and pd.notna(expected_eps):
                    if actual_eps > expected_eps * 1.05:  # 5% beat
                        beat_miss = 'beat'
                    elif actual_eps < expected_eps * 0.95:  # 5% miss
                        beat_miss = 'miss'
                    else:
                        beat_miss = 'meet'

                events.append({
                    'date': date,
                    'event_type': 'earnings',
                    'expected': float(expected_eps) if pd.notna(expected_eps) else 0.0,
                    'actual': float(actual_eps) if pd.notna(actual_eps) else 0.0,
                    'beat_miss': beat_miss,
                    'category': 'tsla'
                })

            result_df = pd.DataFrame(events)

            # Cache for 30 days
            self.cache.set(cache_key, result_df.to_dict('records'))

            print(f"   ✓ Fetched {len(result_df)} earnings events from Alpha Vantage")
            return result_df

        except requests.exceptions.RequestException as e:
            print(f"   ❌ Alpha Vantage API error: {e}")
            return pd.DataFrame()
        except Exception as e:
            print(f"   ❌ Alpha Vantage parsing error: {e}")
            return pd.DataFrame()


class FREDClient:
    """
    FRED (Federal Reserve Economic Data) API client.

    Free tier: Unlimited with free API key
    Fetches: CPI, NFP, and FOMC meeting dates
    """

    def __init__(self, api_key: str = None):
        self.api_key = api_key or config.FRED_API_KEY
        self.cache = APICache()

    def fetch_fomc_dates(self, start_year: int = None) -> pd.DataFrame:
        """
        Fetch FOMC meeting dates from FRED.

        FOMC meetings are tracked via Federal Funds Target Rate changes.
        Series: DFEDTARU (Target Rate Upper Limit)

        Args:
            start_year: Year to start fetching (default: current year)

        Returns:
            DataFrame with FOMC meeting dates
        """
        if not self.api_key:
            print("⚠️  FRED API key not configured. Skipping FOMC fetch.")
            return pd.DataFrame()

        if start_year is None:
            start_year = datetime.now().year

        # Check cache
        cache_key = f"fred_fomc_{start_year}"
        cached = self.cache.get(cache_key, max_age_days=90)
        if cached:
            print(f"   ✓ Using cached FRED FOMC data (age: <90 days)")
            return pd.DataFrame(cached)

        print(f"   📡 Fetching FOMC dates from FRED...")

        try:
            from fredapi import Fred
            fred = Fred(api_key=self.api_key)

            # Fetch Federal Funds Target Rate series
            # Observation dates = FOMC meeting dates
            fomc_series = fred.get_series('DFEDTARU', observation_start=f'{start_year}-01-01')

            events = []
            for date, value in fomc_series.items():
                events.append({
                    'date': pd.Timestamp(date),
                    'event_type': 'fomc',
                    'expected': 0.0,
                    'actual': float(value) if pd.notna(value) else 0.0,
                    'beat_miss': 'neutral',
                    'category': 'macro'
                })

            result_df = pd.DataFrame(events)

            # Cache for 90 days
            self.cache.set(cache_key, result_df.to_dict('records'))

            print(f"   ✓ Fetched {len(result_df)} FOMC dates from FRED")
            return result_df

        except ImportError:
            print("   ❌ fredapi not installed. Run: pip install fredapi")
            return pd.DataFrame()
        except Exception as e:
            print(f"   ❌ FRED API error: {e}")
            return pd.DataFrame()

    def fetch_economic_releases(self) -> pd.DataFrame:
        """
        Fetch CPI and NFP release dates from FRED.

        Note: FRED provides historical data, not future schedules.
        Future dates must be calculated from schedule.

        Returns:
            DataFrame with economic event dates
        """
        if not self.api_key:
            print("⚠️  FRED API key not configured. Skipping FRED fetch.")
            return pd.DataFrame()

        # Check cache (30 day expiry)
        cache_key = "fred_economic_releases"
        cached = self.cache.get(cache_key, max_age_days=30)
        if cached:
            print(f"   ✓ Using cached FRED data (age: <30 days)")
            return pd.DataFrame(cached)

        print(f"   📡 Fetching economic calendar from FRED...")

        try:
            from fredapi import Fred
            fred = Fred(api_key=self.api_key)

            events = []

            # Fetch CPI release dates (series: CPIAUCSL)
            cpi_series = fred.get_series('CPIAUCSL', observation_start='2024-01-01')
            for date, value in cpi_series.items():
                events.append({
                    'date': pd.Timestamp(date),
                    'event_type': 'cpi',
                    'expected': 0.0,
                    'actual': float(value) if pd.notna(value) else 0.0,
                    'beat_miss': 'neutral',
                    'category': 'macro'
                })

            # Fetch NFP/unemployment (series: UNRATE)
            nfp_series = fred.get_series('UNRATE', observation_start='2024-01-01')
            for date, value in nfp_series.items():
                events.append({
                    'date': pd.Timestamp(date),
                    'event_type': 'nfp',
                    'expected': 0.0,
                    'actual': float(value) if pd.notna(value) else 0.0,
                    'beat_miss': 'neutral',
                    'category': 'macro'
                })

            result_df = pd.DataFrame(events)

            # Cache for 30 days
            self.cache.set(cache_key, result_df.to_dict('records'))

            print(f"   ✓ Fetched {len(result_df)} economic events from FRED")
            return result_df

        except ImportError:
            print("   ❌ fredapi not installed. Run: pip install fredapi")
            return pd.DataFrame()
        except Exception as e:
            print(f"   ❌ FRED API error: {e}")
            return pd.DataFrame()


class FOMCScraper:
    """
    Scrape FOMC meeting schedule from Federal Reserve website.

    No API key needed - public data.
    """

    def __init__(self):
        self.base_url = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
        self.cache = APICache()

    def fetch_fomc_schedule(self, year: int = None) -> pd.DataFrame:
        """
        Scrape FOMC meeting dates from Fed website.

        Args:
            year: Year to fetch (default: current year + next year)

        Returns:
            DataFrame with FOMC dates
        """
        if year is None:
            year = datetime.now().year

        # Check cache (90 day expiry - FOMC schedule changes yearly)
        cache_key = f"fomc_schedule_{year}"
        cached = self.cache.get(cache_key, max_age_days=90)
        if cached:
            print(f"   ✓ Using cached FOMC schedule for {year} (age: <90 days)")
            return pd.DataFrame(cached)

        print(f"   📡 Fetching FOMC schedule from Federal Reserve website...")

        try:
            import requests
            from bs4 import BeautifulSoup

            response = requests.get(self.base_url, timeout=30)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')

            # Look for div or table containing meeting dates
            # Federal Reserve site structure may vary - this is a template
            # You may need to adjust selectors based on actual HTML structure

            events = []

            # Placeholder for actual scraping logic
            # This would need to parse the specific HTML structure of the Fed site
            # For now, fall back to predictable schedule

            print(f"   ⚠️  FOMC scraping not fully implemented. Using calculated schedule.")

            # Generate FOMC dates (typically 8 meetings per year)
            # Approximate schedule: Every 6 weeks starting late January
            fomc_dates = self._generate_fomc_schedule(year)

            for date in fomc_dates:
                events.append({
                    'date': pd.Timestamp(date),
                    'event_type': 'fomc',
                    'expected': 0.0,
                    'actual': 0.0,
                    'beat_miss': 'neutral',
                    'category': 'macro'
                })

            result_df = pd.DataFrame(events)

            # Cache for 90 days
            self.cache.set(cache_key, result_df.to_dict('records'))

            print(f"   ✓ Generated {len(result_df)} FOMC dates for {year}")
            return result_df

        except ImportError:
            print("   ❌ beautifulsoup4 not installed. Run: pip install beautifulsoup4")
            return self._generate_fomc_fallback(year)
        except Exception as e:
            print(f"   ⚠️  FOMC fetch error ({e}), using fallback schedule")
            return self._generate_fomc_fallback(year)

    def _generate_fomc_schedule(self, year: int) -> List[datetime]:
        """Generate approximate FOMC meeting dates (8 per year)"""
        # FOMC typically meets 8 times per year
        # Approximate schedule: Late Jan, Mid Mar, Early May, Mid Jun, Late Jul, Mid Sep, Early Nov, Mid Dec
        base_dates = [
            datetime(year, 1, 28),   # Late January
            datetime(year, 3, 15),   # Mid March
            datetime(year, 5, 3),    # Early May
            datetime(year, 6, 14),   # Mid June
            datetime(year, 7, 26),   # Late July
            datetime(year, 9, 20),   # Mid September
            datetime(year, 11, 1),   # Early November
            datetime(year, 12, 13),  # Mid December
        ]
        return base_dates

    def _generate_fomc_fallback(self, year: int) -> pd.DataFrame:
        """Fallback: Generate FOMC schedule from typical pattern"""
        dates = self._generate_fomc_schedule(year)

        events = []
        for date in dates:
            events.append({
                'date': pd.Timestamp(date),
                'event_type': 'fomc',
                'expected': 0.0,
                'actual': 0.0,
                'beat_miss': 'neutral',
                'category': 'macro'
            })

        return pd.DataFrame(events)


class EventScheduleGenerator:
    """
    Generate predictable event schedules (CPI, NFP, Quad Witching).

    These events follow fixed schedules and don't require API calls.
    """

    @staticmethod
    def generate_cpi_schedule(year: int) -> pd.DataFrame:
        """
        Generate CPI release schedule.

        CPI is released around the 12th-15th of each month (for previous month's data).
        """
        events = []

        for month in range(1, 13):
            # CPI typically released 2nd Wednesday of each month
            # Approximate to 13th of each month
            date = datetime(year, month, 13)

            events.append({
                'date': pd.Timestamp(date),
                'event_type': 'cpi',
                'expected': 0.0,
                'actual': 0.0,
                'beat_miss': 'neutral',
                'category': 'macro'
            })

        return pd.DataFrame(events)

    @staticmethod
    def generate_nfp_schedule(year: int) -> pd.DataFrame:
        """
        Generate NFP (Non-Farm Payrolls) release schedule.

        NFP is released first Friday of each month.
        """
        events = []

        for month in range(1, 13):
            # Find first Friday of the month
            first_day = datetime(year, month, 1)
            # Find next Friday (weekday 4)
            days_until_friday = (4 - first_day.weekday()) % 7
            if days_until_friday == 0:
                days_until_friday = 7  # If 1st is Friday, go to next Friday
            first_friday = first_day + timedelta(days=days_until_friday)

            events.append({
                'date': pd.Timestamp(first_friday),
                'event_type': 'nfp',
                'expected': 0.0,
                'actual': 0.0,
                'beat_miss': 'neutral',
                'category': 'macro'
            })

        return pd.DataFrame(events)

    @staticmethod
    def generate_quad_witching_schedule(year: int) -> pd.DataFrame:
        """
        Generate Quadruple Witching schedule.

        Quad Witching: 3rd Friday of March, June, September, December.
        """
        events = []

        for month in [3, 6, 9, 12]:  # Mar, Jun, Sep, Dec
            # Find 3rd Friday
            first_day = datetime(year, month, 1)
            days_until_friday = (4 - first_day.weekday()) % 7
            if days_until_friday == 0:
                days_until_friday = 7
            first_friday = first_day + timedelta(days=days_until_friday)
            third_friday = first_friday + timedelta(days=14)  # +2 weeks

            events.append({
                'date': pd.Timestamp(third_friday),
                'event_type': 'quad_witching',
                'expected': 0.0,
                'actual': 0.0,
                'beat_miss': 'neutral',
                'category': 'macro'
            })

        return pd.DataFrame(events)


def merge_events(existing_csv: pd.DataFrame, new_events: pd.DataFrame) -> pd.DataFrame:
    """
    Merge new events with existing CSV, avoiding duplicates.

    Args:
        existing_csv: Existing events from CSV
        new_events: New events from APIs

    Returns:
        Merged DataFrame with duplicates removed
    """
    if new_events.empty:
        return existing_csv

    # Ensure date columns are datetime
    existing_csv['date'] = pd.to_datetime(existing_csv['date'])
    new_events['date'] = pd.to_datetime(new_events['date'])

    # Concatenate
    merged = pd.concat([existing_csv, new_events], ignore_index=True)

    # Remove duplicates (same date + event_type)
    merged = merged.drop_duplicates(subset=['date', 'event_type'], keep='first')

    # Sort by date
    merged = merged.sort_values('date').reset_index(drop=True)

    return merged


# Testing function
if __name__ == '__main__':
    print("Testing API Fetchers...")

    # Test Alpha Vantage
    print("\n1. Testing Alpha Vantage (TSLA Earnings)...")
    av_client = AlphaVantageClient()
    earnings = av_client.fetch_earnings_calendar('TSLA')
    if not earnings.empty:
        print(f"   Fetched {len(earnings)} earnings events")
        print(earnings.head())

    # Test FRED
    print("\n2. Testing FRED (Economic Data)...")
    fred_client = FREDClient()
    econ_data = fred_client.fetch_economic_releases()
    if not econ_data.empty:
        print(f"   Fetched {len(econ_data)} economic events")
        print(econ_data.head())

    # Test FOMC
    print("\n3. Testing FOMC Schedule...")
    fomc_scraper = FOMCScraper()
    fomc = fomc_scraper.fetch_fomc_schedule(2026)
    print(f"   Generated {len(fomc)} FOMC dates")
    print(fomc)

    # Test Schedule Generators
    print("\n4. Testing Predictable Schedules...")
    generator = EventScheduleGenerator()

    cpi = generator.generate_cpi_schedule(2026)
    print(f"   CPI: {len(cpi)} events")

    nfp = generator.generate_nfp_schedule(2026)
    print(f"   NFP: {len(nfp)} events")

    quad = generator.generate_quad_witching_schedule(2026)
    print(f"   Quad Witching: {len(quad)} events")

    print("\n✅ API Fetchers test complete")
