"""Download 1-min historical bars from IB Gateway.

Usage:
    python -m v15.ib.historical --symbol TSLA --months 6
    python -m v15.ib.historical --symbol TSLA --months 1 --output data/TSLAMin_IB.txt

Output format matches data/TSLAMin.txt:
    YYYYMMDD HHMMSS;open;high;low;close;volume
"""

import argparse
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)


def download_1min_bars(symbol: str, months: int, output: str,
                       host: str, port: int):
    """Download 1-min bars day-by-day from IB and save in TSLAMin.txt format."""
    from v15.ib.client import IBClient

    client = IBClient(host=host, port=port, client_id=99)
    logger.info("Connecting to IB Gateway at %s:%d ...", host, port)
    client.connect()
    logger.info("Connected.")

    # Calculate trading days to fetch
    end_date = datetime.now()
    start_date = end_date - timedelta(days=months * 31)  # overshoot slightly

    all_records = []
    current = end_date
    day_count = 0
    errors = 0

    logger.info("Downloading %s 1-min bars from %s to %s ...",
                symbol, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))

    while current > start_date:
        end_dt_str = current.strftime('%Y%m%d %H:%M:%S') + ' US/Eastern'
        try:
            df = _fetch_one_day(client, symbol, end_dt_str)
            if df is not None and len(df) > 0:
                all_records.append(df)
                day_count += 1
                earliest = df['date'].iloc[0]
                logger.info("  Day %d: %s — %d bars (earliest: %s)",
                             day_count, current.strftime('%Y-%m-%d'),
                             len(df), earliest)
                # Move to previous day (strip tz to keep comparison naive)
                current = pd.Timestamp(earliest).tz_localize(None).to_pydatetime() - timedelta(seconds=1)
            else:
                # No data (weekend/holiday) — just step back 1 day
                current -= timedelta(days=1)
            errors = 0
        except Exception as e:
            errors += 1
            logger.warning("  Error fetching %s: %s", current.strftime('%Y-%m-%d'), e)
            if errors >= 3:
                logger.error("Too many consecutive errors — stopping")
                break
            current -= timedelta(days=1)

        # IB pacing: ~1 request/second
        time.sleep(1.1)

    client.disconnect()

    if not all_records:
        logger.error("No data downloaded!")
        return

    # Combine and sort chronologically
    combined = pd.concat(all_records, ignore_index=True)
    combined = combined.sort_values('date').drop_duplicates(subset='date')
    logger.info("Total: %d bars across %d trading days", len(combined), day_count)

    # Write in TSLAMin.txt format: YYYYMMDD HHMMSS;open;high;low;close;volume
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        for _, row in combined.iterrows():
            dt = pd.Timestamp(row['date'])
            line = (f"{dt.strftime('%Y%m%d %H%M%S')};"
                    f"{row['open']};{row['high']};{row['low']};{row['close']};"
                    f"{int(row['volume'])}")
            f.write(line + '\n')

    logger.info("Saved to %s (%d lines)", output_path, len(combined))


def _fetch_one_day(client, symbol: str, end_dt: str) -> pd.DataFrame:
    """Fetch 1 day of 1-min bars ending at end_dt.

    Uses reqHistoricalData with explicit endDateTime so we can
    page backward through time.
    """
    import asyncio
    from ib_async import Stock

    contract = Stock(symbol, 'SMART', 'USD')

    future = asyncio.run_coroutine_threadsafe(
        client.ib.reqHistoricalDataAsync(
            contract,
            endDateTime=end_dt,
            durationStr='1 D',
            barSizeSetting='1 min',
            whatToShow='TRADES',
            useRTH=False,
            formatDate=1,
        ),
        client._loop,
    )
    bars = future.result(timeout=30)

    if not bars:
        return None

    records = []
    for bar in bars:
        records.append({
            'date': bar.date,
            'open': bar.open,
            'high': bar.high,
            'low': bar.low,
            'close': bar.close,
            'volume': int(bar.volume),
        })
    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(
        description='Download 1-min historical bars from IB Gateway')
    parser.add_argument('--symbol', default='TSLA', help='Symbol to download')
    parser.add_argument('--months', type=int, default=6,
                        help='Number of months of history')
    parser.add_argument('--output', default=None,
                        help='Output file path (default: data/{SYMBOL}Min_IB.txt)')
    parser.add_argument('--host', default='127.0.0.1', help='IB Gateway host')
    parser.add_argument('--port', type=int, default=4002, help='IB Gateway port')

    args = parser.parse_args()

    if args.output is None:
        args.output = f'data/{args.symbol}Min_IB.txt'

    download_1min_bars(
        symbol=args.symbol,
        months=args.months,
        output=args.output,
        host=args.host,
        port=args.port,
    )


if __name__ == '__main__':
    main()
