"""Download historical trade ticks from IB Gateway and save as per-day Parquet files.

Usage:
    python -m v15.ib.tick_downloader --symbol TSLA --start 2025-01-01 --end 2025-03-01
    python -m v15.ib.tick_downloader --symbol TSLA --start 2025-01-01 --end 2025-03-01 --verify-only
    python -m v15.ib.tick_downloader --symbol TSLA --start 2025-01-01 --end 2025-03-01 --redownload 2025-01-15

Output: data/ticks/{SYMBOL}/YYYY-MM-DD.parquet (one file per trading day)
"""

import argparse
import asyncio
import logging
import time as _time
from collections import deque
from datetime import datetime, date, timedelta
from pathlib import Path

import numpy as np
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
REQUIRED_COLUMNS = ['time', 'price', 'size', 'exchange', 'conditions',
                    'past_limit', 'unreported', 'seq']

# Known US market early-close dates (month, day) → session ends 17:00 ET instead of 20:00
# These are approximate — Jul 3 (day before Independence Day), day after Thanksgiving,
# Christmas Eve. Actual dates shift by year.
_EARLY_CLOSE_MONTH_DAYS = {(7, 3), (11, 28), (11, 29), (12, 24)}

# Extended session boundaries (ET, tz-naive)
SESSION_START_ET = '04:00:00'
SESSION_END_NORMAL_ET = '19:59:00'  # Last included minute for normal days
SESSION_END_EARLY_ET = '16:59:00'   # Last included minute for early close days


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
# are variable — add known ones
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


def _session_end_minute(d: date) -> str:
    """Return the last included minute timestamp for the session (ET, tz-naive)."""
    if _is_early_close(d):
        return f'{d.isoformat()} {SESSION_END_EARLY_ET}'
    return f'{d.isoformat()} {SESSION_END_NORMAL_ET}'


class RateLimiter:
    """Triple rolling-window rate limiter for IB historical tick pacing."""

    def __init__(self):
        # Rule 1: 60 requests per 10 minutes
        self._window_10m: deque = deque()
        # Rule 2: No identical request within 15s (handled by always advancing startDateTime)
        # Rule 3: No 6+ requests for same contract/type within 2s
        self._window_2s: deque = deque()

    def wait(self):
        """Sleep until all pacing rules clear."""
        now = _time.time()

        # Rule 1: 60 per 10 minutes
        while len(self._window_10m) >= 58:  # Leave 2-request margin
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


def _fetch_ticks_page(client, contract, start_dt_str: str,
                      num_ticks: int = 1000) -> list:
    """Fetch one page of historical trade ticks from IB."""
    async def _req():
        return await client.ib.reqHistoricalTicksAsync(
            contract,
            startDateTime=start_dt_str,
            endDateTime='',
            numberOfTicks=num_ticks,
            whatToShow='TRADES',
            useRth=False,
            ignoreSize=False,
        )

    future = asyncio.run_coroutine_threadsafe(_req(), client._loop)
    return future.result(timeout=60)


def download_ticks_for_day(client, contract, day: date, output_dir: Path,
                           rate_limiter: RateLimiter) -> dict:
    """Download all trade ticks for one trading day. Returns status dict."""
    final_path = output_dir / f'{day.isoformat()}.parquet'
    tmp_path = output_dir / f'{day.isoformat()}.parquet.tmp'

    if final_path.exists():
        return {'status': 'skipped', 'day': day, 'reason': 'already exists'}

    # Clean up any stale temp file
    if tmp_path.exists():
        tmp_path.unlink()

    session_end_str = _session_end_minute(day)
    session_end_ts = pd.Timestamp(session_end_str)

    all_ticks = []
    seq_counter = 0
    previous_max_time = None
    start_dt = f'{day.strftime("%Y%m%d")} {SESSION_START_ET} US/Eastern'
    page_count = 0

    while True:
        rate_limiter.wait()
        page_count += 1

        try:
            ticks = _fetch_ticks_page(client, contract, start_dt)
        except Exception as e:
            logger.warning("  Error fetching ticks for %s page %d: %s",
                           day, page_count, e)
            return {'status': 'error', 'day': day, 'reason': str(e)}

        if not ticks:
            break

        # Convert to records
        page_records = []
        for tick in ticks:
            tick_time = pd.Timestamp(tick.time)
            # Strip timezone if present
            if tick_time.tzinfo is not None:
                tick_time = tick_time.tz_convert('US/Eastern').tz_localize(None)

            past_limit = bool(getattr(tick.tickAttribLast, 'pastLimit', False)
                              if hasattr(tick, 'tickAttribLast') and tick.tickAttribLast
                              else False)
            unreported = bool(getattr(tick.tickAttribLast, 'unreported', False)
                              if hasattr(tick, 'tickAttribLast') and tick.tickAttribLast
                              else False)

            page_records.append({
                'time': tick_time,
                'price': float(tick.price),
                'size': int(tick.size),
                'exchange': str(getattr(tick, 'exchange', '')),
                'conditions': str(getattr(tick, 'specialConditions', '') or ''),
                'past_limit': past_limit,
                'unreported': unreported,
                'seq': seq_counter,
            })
            seq_counter += 1

        if not page_records:
            break

        max_time = max(r['time'] for r in page_records)

        # Forward progress assertion
        if previous_max_time is not None and max_time <= previous_max_time:
            # Retry once
            logger.warning("  Non-advancing page for %s (stuck at %s). Retrying...",
                           day, max_time)
            rate_limiter.wait()
            try:
                ticks_retry = _fetch_ticks_page(client, contract, start_dt)
            except Exception:
                ticks_retry = []

            if ticks_retry:
                retry_max = max(pd.Timestamp(t.time) if not pd.Timestamp(t.time).tzinfo
                                else pd.Timestamp(t.time).tz_convert('US/Eastern').tz_localize(None)
                                for t in ticks_retry)
                if retry_max > previous_max_time:
                    # Retry succeeded — reprocess (simplified: just continue)
                    pass
                else:
                    return {'status': 'error', 'day': day,
                            'reason': f'Non-advancing page stuck at {max_time}'}
            else:
                return {'status': 'error', 'day': day,
                        'reason': f'Non-advancing page stuck at {max_time}'}

        all_ticks.extend(page_records)
        previous_max_time = max_time

        # Check if we've passed session close
        if max_time >= session_end_ts:
            break

        # Advance cursor by +1 second
        next_start = max_time + pd.Timedelta(seconds=1)
        start_dt = next_start.strftime('%Y%m%d %H:%M:%S') + ' US/Eastern'

    if not all_ticks:
        return {'status': 'error', 'day': day, 'reason': 'No ticks returned'}

    # Build DataFrame
    df = pd.DataFrame(all_ticks)

    # Check for duplicates (should be impossible with +1s pagination)
    dup_mask = df.duplicated(subset=['time', 'price', 'size', 'exchange'], keep=False)
    if dup_mask.any():
        return {'status': 'error', 'day': day,
                'reason': f'Found {dup_mask.sum()} duplicate ticks — pagination bug'}

    # Validate
    validation = _validate_day(df, day, session_end_ts)
    if validation['error']:
        return {'status': 'error', 'day': day, 'reason': validation['error']}

    # Write to temp file
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, tmp_path, compression='snappy')

    # Promote to final
    tmp_path.rename(final_path)

    return {'status': 'ok', 'day': day, 'ticks': len(df), 'pages': page_count,
            'warnings': validation.get('warnings', [])}


def _validate_day(df: pd.DataFrame, day: date,
                  session_end_ts: pd.Timestamp) -> dict:
    """Validate a day's tick data. Returns {'error': str|None, 'warnings': list}."""
    warnings = []

    # 1. Non-empty (at least 100 ticks)
    if len(df) < 100:
        return {'error': f'Only {len(df)} ticks (minimum 100)', 'warnings': warnings}

    # 2. Monotonic timestamps (allow equal)
    if not df['time'].is_monotonic_increasing:
        return {'error': 'Timestamps not monotonically increasing', 'warnings': warnings}

    # 3. Price sanity
    prices = df['price'].values
    if (prices < 0).any():
        return {'error': 'Negative prices found', 'warnings': warnings}
    # Check for >10x jumps (excluding zero-price halt markers)
    nonzero = prices[prices > 0]
    if len(nonzero) > 1:
        ratios = nonzero[1:] / nonzero[:-1]
        if (ratios > 10).any() or (ratios < 0.1).any():
            return {'error': 'Price jump > 10x between consecutive ticks',
                    'warnings': warnings}

    # 4. Zero-price handling (check flags)
    zero_mask = df['price'] == 0
    if zero_mask.any():
        zero_no_flag = zero_mask & ~df['past_limit'] & ~df['unreported']
        if zero_no_flag.any():
            warnings.append(f'{zero_no_flag.sum()} zero-price ticks without halt/unreported flags')

    # 5. Date ownership
    tick_dates = df['time'].dt.date
    wrong_date = tick_dates != day
    if wrong_date.any():
        return {'error': f'{wrong_date.sum()} ticks belong to wrong date',
                'warnings': warnings}

    # 6. Session coverage — last tick must be in final minute
    last_tick_time = df['time'].iloc[-1]
    if last_tick_time < session_end_ts:
        return {'error': f'Truncated: last tick at {last_tick_time}, '
                         f'expected >= {session_end_ts}',
                'warnings': warnings}

    # 7. Seq monotonicity
    if not df['seq'].is_monotonic_increasing:
        return {'error': 'Seq not monotonically increasing', 'warnings': warnings}

    return {'error': None, 'warnings': warnings}


def verify_tick_files(tick_dir: Path, symbol: str, start: date, end: date):
    """Verify existing tick Parquet files for completeness and integrity."""
    logger.info("Verifying tick files in %s for %s (%s to %s)",
                tick_dir, symbol, start, end)

    total_days = 0
    ok_days = 0
    missing_days = 0
    error_days = 0
    total_ticks = 0

    d = start
    while d <= end:
        if not _is_trading_day(d):
            d += timedelta(days=1)
            continue

        total_days += 1
        path = tick_dir / f'{d.isoformat()}.parquet'

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

            session_end_ts = pd.Timestamp(_session_end_minute(d))
            validation = _validate_day(df, d, session_end_ts)
            if validation['error']:
                logger.error("  INVALID %s: %s", d, validation['error'])
                error_days += 1
            else:
                ok_days += 1
                total_ticks += len(df)
                for w in validation.get('warnings', []):
                    logger.warning("  WARNING %s: %s", d, w)

        except Exception as e:
            logger.error("  CORRUPT %s: %s", d, e)
            error_days += 1

        d += timedelta(days=1)

    logger.info("\nSummary: %d trading days, %d OK, %d missing, %d errors, %d total ticks",
                total_days, ok_days, missing_days, error_days, total_ticks)
    return {'total': total_days, 'ok': ok_days, 'missing': missing_days,
            'errors': error_days, 'ticks': total_ticks}


def download_ticks(symbol: str, start: date, end: date, output_dir: str,
                   host: str, port: int, redownload: str = None):
    """Download trade ticks for a date range."""
    from v15.ib.client import IBClient
    from ib_async import Stock

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    client = IBClient(host=host, port=port, client_id=98)
    logger.info("Connecting to IB Gateway at %s:%d ...", host, port)
    client.connect()
    logger.info("Connected.")

    contract = Stock(symbol, 'SMART', 'USD')
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
    total_ticks = 0

    while d <= end:
        if not _is_trading_day(d):
            d += timedelta(days=1)
            continue

        result = download_ticks_for_day(client, contract, d, out_path, rate_limiter)

        if result['status'] == 'ok':
            days_ok += 1
            total_ticks += result['ticks']
            logger.info("  %s: %d ticks (%d pages)%s",
                        d, result['ticks'], result['pages'],
                        f" [{', '.join(result['warnings'])}]" if result.get('warnings') else '')
        elif result['status'] == 'skipped':
            days_skipped += 1
            logger.debug("  %s: skipped (%s)", d, result['reason'])
        else:
            days_error += 1
            logger.error("  %s: ERROR — %s", d, result['reason'])

            # Retry up to 3 times for expected trading days
            for retry in range(3):
                logger.info("  Retrying %s (attempt %d/3)...", d, retry + 1)
                # Remove any temp file
                tmp = out_path / f'{d.isoformat()}.parquet.tmp'
                if tmp.exists():
                    tmp.unlink()
                _time.sleep(5)
                result = download_ticks_for_day(client, contract, d, out_path,
                                                rate_limiter)
                if result['status'] == 'ok':
                    days_ok += 1
                    days_error -= 1
                    total_ticks += result['ticks']
                    logger.info("  %s: retry succeeded — %d ticks", d, result['ticks'])
                    break
            else:
                logger.error("  %s: FAILED after 3 retries", d)

        d += timedelta(days=1)

    client.disconnect()
    logger.info("\nDone: %d OK, %d skipped, %d errors, %d total ticks",
                days_ok, days_skipped, days_error, total_ticks)


def main():
    parser = argparse.ArgumentParser(
        description='Download historical trade ticks from IB Gateway')
    parser.add_argument('--symbol', default='TSLA', help='Symbol to download')
    parser.add_argument('--start', required=True, help='Start date YYYY-MM-DD')
    parser.add_argument('--end', required=True, help='End date YYYY-MM-DD')
    parser.add_argument('--output', default=None,
                        help='Output directory (default: data/ticks/{SYMBOL})')
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
        args.output = f'data/ticks/{args.symbol}'

    if args.verify_only:
        verify_tick_files(Path(args.output), args.symbol, start, end)
    else:
        download_ticks(
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
