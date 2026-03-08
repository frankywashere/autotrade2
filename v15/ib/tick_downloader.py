"""Download historical 5-second bars from IB Gateway and save as per-day Parquet files.

Usage:
    python -m v15.ib.tick_downloader --symbol TSLA --start 2025-01-01 --end 2026-03-06
    python -m v15.ib.tick_downloader --symbol TSLA --start 2025-01-01 --end 2026-03-06 --verify-only
    python -m v15.ib.tick_downloader --symbol TSLA --start 2025-01-01 --end 2026-03-06 --redownload 2025-01-15

Output: data/bars_5s/{SYMBOL}/YYYY-MM-DD.parquet (one file per trading day)

Each file contains 5-second OHLCV bars for the full extended session (04:00-20:00 ET).
Bars are fetched in 1-hour chunks (3600 S duration, ~720 bars per chunk, 16 chunks per day).
"""

import argparse
import asyncio
import logging
import time as _time
from collections import deque
from datetime import datetime, date, timedelta
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

# Required Parquet schema columns
REQUIRED_COLUMNS = ['time', 'open', 'high', 'low', 'close', 'volume']

# Known US market early-close dates (month, day) → session ends 17:00 ET instead of 20:00
_EARLY_CLOSE_MONTH_DAYS = {(7, 3), (11, 28), (11, 29), (12, 24)}

# Extended session boundaries (ET hours, 0-23)
SESSION_START_HOUR = 4   # 04:00 ET
SESSION_END_HOUR_NORMAL = 20  # 20:00 ET (last chunk ends here)
SESSION_END_HOUR_EARLY = 17   # 17:00 ET for early close days


def _is_weekend(d: date) -> bool:
    return d.weekday() >= 5


# Known US market holidays (non-exhaustive, covers 2024-2027)
_US_HOLIDAYS = set()
for y in range(2024, 2028):
    _US_HOLIDAYS.update([
        date(y, 1, 1),    # New Year's Day
        date(y, 7, 4),    # Independence Day
        date(y, 12, 25),  # Christmas Day
    ])
# MLK, Presidents', Good Friday, Memorial, Juneteenth, Labor, Thanksgiving
_US_HOLIDAYS.update([
    date(2025, 1, 20), date(2025, 2, 17), date(2025, 4, 18),
    date(2025, 5, 26), date(2025, 6, 19), date(2025, 9, 1),
    date(2025, 11, 27),
    date(2026, 1, 19), date(2026, 2, 16), date(2026, 4, 3),
    date(2026, 5, 25), date(2026, 6, 19), date(2026, 9, 7),
    date(2026, 11, 26),
])


def _is_trading_day(d: date) -> bool:
    if _is_weekend(d):
        return False
    if d in _US_HOLIDAYS:
        return False
    return True


def _is_early_close(d: date) -> bool:
    return (d.month, d.day) in _EARLY_CLOSE_MONTH_DAYS


def _session_end_hour(d: date) -> int:
    if _is_early_close(d):
        return SESSION_END_HOUR_EARLY
    return SESSION_END_HOUR_NORMAL


class RateLimiter:
    """Rolling-window rate limiter for IB historical data pacing."""

    def __init__(self):
        # Rule 1: 60 requests per 10 minutes
        self._window_10m: deque = deque()
        # Rule 3: No 6+ requests for same contract/type within 2s
        self._window_2s: deque = deque()

    def wait(self):
        """Sleep until all pacing rules clear."""
        now = _time.time()

        # Rule 1: 60 per 10 minutes (leave 2-request margin)
        while len(self._window_10m) >= 58:
            oldest = self._window_10m[0]
            wait = oldest + 600 - now
            if wait > 0:
                logger.debug("  Pacing: waiting %.1fs (60/10min rule)", wait)
                _time.sleep(wait + 0.1)
                now = _time.time()
            self._window_10m.popleft()

        # Rule 3: < 6 requests in 2 seconds
        cutoff_2s = now - 2.0
        while self._window_2s and self._window_2s[0] < cutoff_2s:
            self._window_2s.popleft()
        if len(self._window_2s) >= 5:
            wait = self._window_2s[0] + 2.0 - now
            if wait > 0:
                _time.sleep(wait + 0.1)

        # Record this request
        now = _time.time()
        self._window_10m.append(now)
        self._window_2s.append(now)


def _fetch_bars_chunk(client, contract, end_dt_str: str) -> list:
    """Fetch one 1-hour chunk of 5-second bars from IB.

    Uses reqHistoricalDataAsync which returns a coroutine in ib_async,
    bridged to the main thread via run_coroutine_threadsafe.

    Args:
        client: IBClient instance
        contract: Qualified IB contract
        end_dt_str: End datetime string like '20250102 05:00:00 US/Eastern'

    Returns:
        List of ib_async BarData objects
    """
    future = asyncio.run_coroutine_threadsafe(
        client.ib.reqHistoricalDataAsync(
            contract,
            endDateTime=end_dt_str,
            durationStr='3600 S',
            barSizeSetting='5 secs',
            whatToShow='TRADES',
            useRTH=False,
            formatDate=1,
        ),
        client._loop,
    )
    bars = future.result(timeout=60)
    return bars or []


def download_bars_for_day(client, contract, day: date, output_dir: Path,
                          rate_limiter: RateLimiter) -> dict:
    """Download all 5-second bars for one trading day. Returns status dict."""
    final_path = output_dir / f'{day.isoformat()}.parquet'
    tmp_path = output_dir / f'{day.isoformat()}.parquet.tmp'

    if final_path.exists():
        return {'status': 'skipped', 'day': day, 'reason': 'already exists'}

    # Clean up any stale temp file
    if tmp_path.exists():
        tmp_path.unlink()

    end_hour = _session_end_hour(day)
    all_records = []
    chunk_count = 0

    # Loop over 1-hour chunks: 04:00→05:00, 05:00→06:00, ..., 19:00→20:00
    for hour_end in range(SESSION_START_HOUR + 1, end_hour + 1):
        rate_limiter.wait()
        chunk_count += 1

        end_dt = f'{day.strftime("%Y%m%d")} {hour_end:02d}:00:00 US/Eastern'
        logger.info("  %s chunk %d/%d (end=%02d:00)...",
                    day, chunk_count, end_hour - SESSION_START_HOUR, hour_end)

        try:
            bars = _fetch_bars_chunk(client, contract, end_dt)
        except Exception as e:
            logger.warning("  Error fetching %s chunk %d: %s", day, chunk_count, e)
            return {'status': 'error', 'day': day, 'reason': str(e)}

        for bar in bars:
            bar_time = pd.Timestamp(bar.date)
            # Strip timezone if present
            if bar_time.tzinfo is not None:
                bar_time = bar_time.tz_convert('US/Eastern').tz_localize(None)

            all_records.append({
                'time': bar_time,
                'open': float(bar.open),
                'high': float(bar.high),
                'low': float(bar.low),
                'close': float(bar.close),
                'volume': int(bar.volume),
            })

    if not all_records:
        return {'status': 'error', 'day': day, 'reason': 'No bars returned'}

    # Build DataFrame and deduplicate (overlapping chunk boundaries)
    df = pd.DataFrame(all_records)
    df = df.drop_duplicates(subset=['time'], keep='first')
    df = df.sort_values('time').reset_index(drop=True)

    # Filter to only this day's bars
    day_start = pd.Timestamp(f'{day.isoformat()} 00:00:00')
    day_end = pd.Timestamp(f'{day.isoformat()} 23:59:59')
    df = df[(df['time'] >= day_start) & (df['time'] <= day_end)]

    if df.empty:
        return {'status': 'error', 'day': day, 'reason': 'No bars after filtering'}

    # Validate
    validation = _validate_day(df, day)
    if validation['error']:
        return {'status': 'error', 'day': day, 'reason': validation['error']}

    # Write to temp file then promote
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, tmp_path, compression='snappy')
    tmp_path.rename(final_path)

    return {'status': 'ok', 'day': day, 'bars': len(df), 'chunks': chunk_count,
            'warnings': validation.get('warnings', [])}


def _validate_day(df: pd.DataFrame, day: date) -> dict:
    """Validate a day's 5-second bar data. Returns {'error': str|None, 'warnings': list}."""
    warnings = []

    # 1. Non-empty (at least 10 bars — could be a very light pre-market day)
    if len(df) < 10:
        return {'error': f'Only {len(df)} bars (minimum 10)', 'warnings': warnings}

    # 2. Monotonic timestamps
    if not df['time'].is_monotonic_increasing:
        return {'error': 'Timestamps not monotonically increasing', 'warnings': warnings}

    # 3. Price sanity — no negatives, no >10x jumps
    prices = df['close'].values
    if (prices <= 0).any():
        return {'error': 'Zero or negative close prices found', 'warnings': warnings}
    ratios = prices[1:] / prices[:-1]
    if (ratios > 10).any() or (ratios < 0.1).any():
        return {'error': 'Price jump > 10x between consecutive bars', 'warnings': warnings}

    # 4. OHLC consistency
    if (df['high'] < df['low']).any():
        return {'error': 'high < low found', 'warnings': warnings}
    if (df['high'] < df['open']).any() or (df['high'] < df['close']).any():
        warnings.append('high < open or high < close in some bars')
    if (df['low'] > df['open']).any() or (df['low'] > df['close']).any():
        warnings.append('low > open or low > close in some bars')

    # 5. Date ownership
    tick_dates = df['time'].dt.date
    wrong_date = tick_dates != day
    if wrong_date.any():
        return {'error': f'{wrong_date.sum()} bars belong to wrong date',
                'warnings': warnings}

    # 6. RTH coverage — should have bars during 9:30-16:00
    rth_bars = df[(df['time'].dt.hour >= 10) & (df['time'].dt.hour < 16)]
    if len(rth_bars) < 100:
        warnings.append(f'Only {len(rth_bars)} RTH bars (expected ~4600+)')

    return {'error': None, 'warnings': warnings}


def verify_bar_files(bar_dir: Path, symbol: str, start: date, end: date):
    """Verify existing 5-second bar Parquet files for completeness and integrity."""
    logger.info("Verifying bar files in %s for %s (%s to %s)",
                bar_dir, symbol, start, end)

    total_days = 0
    ok_days = 0
    missing_days = 0
    error_days = 0
    total_bars = 0

    d = start
    while d <= end:
        if not _is_trading_day(d):
            d += timedelta(days=1)
            continue

        total_days += 1
        path = bar_dir / f'{d.isoformat()}.parquet'

        if not path.exists():
            logger.warning("  MISSING: %s", d)
            missing_days += 1
            d += timedelta(days=1)
            continue

        try:
            df = pd.read_parquet(path)

            # Schema check
            missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)
            if missing_cols:
                logger.error("  SCHEMA ERROR %s: missing columns %s", d, missing_cols)
                error_days += 1
                d += timedelta(days=1)
                continue

            validation = _validate_day(df, d)
            if validation['error']:
                logger.error("  INVALID %s: %s", d, validation['error'])
                error_days += 1
            else:
                ok_days += 1
                total_bars += len(df)
                for w in validation.get('warnings', []):
                    logger.warning("  WARNING %s: %s", d, w)

        except Exception as e:
            logger.error("  CORRUPT %s: %s", d, e)
            error_days += 1

        d += timedelta(days=1)

    logger.info("\nSummary: %d trading days, %d OK, %d missing, %d errors, %d total bars",
                total_days, ok_days, missing_days, error_days, total_bars)
    return {'total': total_days, 'ok': ok_days, 'missing': missing_days,
            'errors': error_days, 'bars': total_bars}


def download_bars(symbol: str, start: date, end: date, output_dir: str,
                  host: str, port: int, redownload: str = None):
    """Download 5-second bars for a date range."""
    from v15.ib.client import IBClient
    from ib_async import Stock

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    client = IBClient(host=host, port=port, client_id=98)
    logger.info("Connecting to IB Gateway at %s:%d ...", host, port)
    client.connect()
    logger.info("Connected.")

    contract = Stock(symbol, 'SMART', 'USD')
    # Qualify contract (required for historical data requests)
    qf = asyncio.run_coroutine_threadsafe(
        client.ib.qualifyContractsAsync(contract), client._loop)
    qf.result(timeout=15)
    logger.info("Contract qualified: %s (conId=%d)", symbol, contract.conId)
    rate_limiter = RateLimiter()

    # If redownloading specific day, remove its file first
    if redownload:
        rd = datetime.strptime(redownload, '%Y-%m-%d').date()
        rd_path = out_path / f'{rd.isoformat()}.parquet'
        if rd_path.exists():
            rd_path.unlink()
            logger.info("Removed %s for re-download", rd_path)

    d = start
    days_ok = 0
    days_error = 0
    days_skipped = 0
    total_bars = 0

    while d <= end:
        if not _is_trading_day(d):
            d += timedelta(days=1)
            continue

        result = download_bars_for_day(client, contract, d, out_path, rate_limiter)

        if result['status'] == 'ok':
            days_ok += 1
            total_bars += result['bars']
            logger.info("  %s: %d bars (%d chunks)%s",
                        d, result['bars'], result['chunks'],
                        f" [{', '.join(result['warnings'])}]" if result.get('warnings') else '')
        elif result['status'] == 'skipped':
            days_skipped += 1
            logger.debug("  %s: skipped (%s)", d, result['reason'])
        else:
            days_error += 1
            logger.error("  %s: ERROR — %s", d, result['reason'])

            # Retry up to 3 times
            for retry in range(3):
                logger.info("  Retrying %s (attempt %d/3)...", d, retry + 1)
                tmp = out_path / f'{d.isoformat()}.parquet.tmp'
                if tmp.exists():
                    tmp.unlink()
                _time.sleep(5)
                result = download_bars_for_day(client, contract, d, out_path,
                                               rate_limiter)
                if result['status'] == 'ok':
                    days_ok += 1
                    days_error -= 1
                    total_bars += result['bars']
                    logger.info("  %s: retry succeeded — %d bars", d, result['bars'])
                    break
            else:
                logger.error("  %s: FAILED after 3 retries", d)

        d += timedelta(days=1)

    client.disconnect()
    logger.info("\nDone: %d OK, %d skipped, %d errors, %d total bars",
                days_ok, days_skipped, days_error, total_bars)


def main():
    parser = argparse.ArgumentParser(
        description='Download historical 5-second bars from IB Gateway')
    parser.add_argument('--symbol', default='TSLA', help='Symbol to download')
    parser.add_argument('--start', required=True, help='Start date YYYY-MM-DD')
    parser.add_argument('--end', required=True, help='End date YYYY-MM-DD')
    parser.add_argument('--output', default=None,
                        help='Output directory (default: data/bars_5s/{SYMBOL})')
    parser.add_argument('--host', default='192.168.0.152', help='IB Gateway host')
    parser.add_argument('--port', type=int, default=4002, help='IB Gateway port')
    parser.add_argument('--verify-only', action='store_true',
                        help='Only verify existing files, do not download')
    parser.add_argument('--redownload', default=None,
                        help='Re-download specific date YYYY-MM-DD')

    args = parser.parse_args()

    start = datetime.strptime(args.start, '%Y-%m-%d').date()
    end = datetime.strptime(args.end, '%Y-%m-%d').date()

    if args.output is None:
        args.output = f'data/bars_5s/{args.symbol}'

    if args.verify_only:
        verify_bar_files(Path(args.output), args.symbol, start, end)
    else:
        download_bars(
            symbol=args.symbol,
            start=start,
            end=end,
            output_dir=args.output,
            host=args.host,
            port=args.port,
            redownload=args.redownload,
        )


if __name__ == '__main__':
    main()
