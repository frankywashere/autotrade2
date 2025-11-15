from pathlib import Path

import pandas as pd
import yfinance as yf


def download_price_data(
    symbol: str,
    start: str,
    end: str,
    interval: str,
    out_dir: Path,
) -> Path:
    """
    Download OHLCV data via yfinance and save to CSV.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{symbol}_{interval}_{start}_{end}.csv".replace(":", "-")
    out_path = out_dir / fname

    data = yf.download(
        symbol,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=True,
        progress=False,
    )
    if data.empty:
        raise RuntimeError(f"No data downloaded for {symbol} {interval} {start} {end}")

    data.reset_index(inplace=True)
    data.rename(columns={"Datetime": "timestamp", "Date": "timestamp"}, inplace=True)
    data["timestamp"] = pd.to_datetime(data["timestamp"]).dt.strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    data.to_csv(out_path, index=False)
    return out_path


