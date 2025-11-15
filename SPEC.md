Alternator Trading Platform - System Specification
==================================================

Overview
--------

Alternator is a research-grade trading platform scaffold focused on hierarchical time-series modeling and high-confidence trade tracking.

Core goals:
- Ingest 1-minute (or other interval) OHLCV data for symbols like TSLA/SPY.
- Train a hierarchical time-series model on sequences of bars.
- Backtest the model, logging all predictions and high-confidence trade setups to SQLite.
- Expose a simple web dashboard to visualize predictions/trades.
- Send Telegram alerts for high-confidence trades.

High-Level Architecture
-----------------------

Project layout (key files only):

- `config/`
  - `config.yaml` – global configuration (paths, training defaults, alerts, server).
  - `api_keys.json` – Telegram bot token and chat ID.
- `src/alternator/`
  - `config.py` – config loader and dataclasses.
  - `data/`
    - `ingest.py` – yfinance-based data downloader.
    - `dataset.py` – PyTorch `Dataset` for OHLCV sequences.
  - `model/`
    - `hierarchical_model.py` – hierarchical liquid neural network model.
  - `training/`
    - `trainer.py` – training loop.
    - `backtester.py` – backtesting loop with DB logging and Telegram alerts.
  - `db/`
    - `schema.py` – SQLite database schema creation.
    - `operations.py` – CRUD wrappers for predictions and trades.
  - `alerts/`
    - `telegram_client.py` – Telegram client for alerts.
  - `cli/`
    - `menu.py` – interactive CLI menu.
  - `server/`
    - `app.py` – FastAPI application.
    - `templates/index.html` – HTML dashboard.
    - `static/style.css` – basic styling.
- `run_cli.py` – entry script for the interactive CLI.
- `run_server.py` – entry script to start the web server.

Configuration
-------------

`config/config.yaml`:

- `paths.data_dir` – directory for CSV price data.
- `paths.models_dir` – directory for saved model weights.
- `paths.db_dir` – directory for SQLite DB files.
- `training.*` – defaults for symbol, sequence length, batch size, learning rate, epochs.
- `backtest.*` – default test period (currently informational).
- `alerts.confidence_threshold` – minimum confidence for a prediction to be logged as a high-confidence trade.
- `server.host` / `server.port` – web server binding.

`config/api_keys.json`:

- `telegram.bot_token` – Telegram bot token.
- `telegram.chat_id` – Telegram chat ID.

Databases
---------

All DB files live in `db/` (configurable via `paths.db_dir`).

- `predictions.db`
  - Table `predictions`:
    - `id` INTEGER PRIMARY KEY
    - `timestamp` TEXT (ISO-8601)
    - `symbol` TEXT
    - `timeframe` TEXT
    - `predicted_high` REAL
    - `predicted_low` REAL
    - `confidence` REAL
    - `model_name` TEXT
    - `extra` JSON (arbitrary metadata)

- `high_confidence_trades.db`
  - Table `high_confidence_trades`:
    - `id` INTEGER PRIMARY KEY
    - `timestamp` TEXT
    - `symbol` TEXT
    - `direction` TEXT (`"long"` or `"short"`)
    - `timeframe` TEXT
    - `entry_price` REAL
    - `target_price` REAL
    - `stop_price` REAL
    - `max_hold_time_minutes` INTEGER
    - `confidence` REAL
    - `expected_return_pct` REAL
    - `rationale` TEXT (JSON-encoded)
    - `status` TEXT (default `"pending"`)

Databases are initialized via `alternator.db.init_databases(db_dir)` and called automatically by the CLI and server.

Model
-----

File: `src/alternator/model/hierarchical_model.py`

- `HierarchicalTimeSeriesModel`:
  - Input: `[batch, seq_len, input_size]` where `input_size=5` (OHLCV).
  - Fast layer: liquid layer (`LiquidLayer`) over the full sequence, built from `LiquidCell`s.
  - Medium layer: liquid layer over a pooled representation of fast outputs.
  - Slow layer: liquid layer over a pooled representation of medium outputs.
  - Liquid cell update:
    - `h_new = h + (1 / tau(x, h)) * (g(x, h) - h)` where `tau(x, h)` is a learned positive time constant and `g` is a nonlinear transform.
  - Fusion: concatenation of final hidden states from fast/medium/slow.
  - Head: fully-connected MLP producing `[pred_high, pred_low, raw_confidence]`.
  - `predict()` applies `sigmoid` to the confidence logit and returns a dict:
    - `"high"` – tensor of predicted highs.
    - `"low"` – tensor of predicted lows.
    - `"confidence"` – tensor in `[0, 1]`.

Data Pipeline
-------------

**Ingestion (`alternator.data.ingest.download_price_data`)**

- Uses yfinance to download OHLCV data for a given:
  - `symbol`
  - `start`, `end` (YYYY-MM-DD)
  - `interval` (e.g., `1m`, `15m`, `1h`, `1d`)
- Saves CSV to `data/` (or configured `data_dir`) with columns:
  - `timestamp`, `Open`, `High`, `Low`, `Close`, `Volume`

**Dataset (`alternator.data.dataset.TimeSeriesDataset`)**

- Loads a single CSV.
- Builds:
  - `features = [Open, High, Low, Close, Volume]`
  - `targets = [High, Low]` (next bar of the sequence).
- For index `i`, returns:
  - `x = features[i : i + sequence_length]`
  - `y = targets[i + sequence_length - 1]`

Training
--------

File: `src/alternator/training/trainer.py`

- `train_model(app_config, csv_path, model_out_path=None, device="cpu")`
  - Uses `TimeSeriesDataset` with `sequence_length` from config.
  - Dataloader with `batch_size`, `shuffle=True`.
  - Model: new `HierarchicalTimeSeriesModel`.
  - Loss: MSE on high/low prediction.
  - Optimizer: Adam.
  - Iterates for `epochs`.
  - Saves final `state_dict()` to:
    - `models/hierarchical_model.pth` by default.

Backtesting
-----------

File: `src/alternator/training/backtester.py`

- `backtest_model(app_config, csv_path, model_path, device="cpu", model_name="hierarchical")`
  - Loads trained `HierarchicalTimeSeriesModel` from `model_path`.
  - Iterates through the dataset one sample at a time (batch size 1).
  - For each window:
    - Runs `model.predict()`.
    - Logs a `predictions` row in `predictions.db`.
    - If `confidence >= alerts.confidence_threshold`:
      - Derives:
        - `direction`: `"long"` if predicted high > current price, else `"short"`.
        - `expected_return_pct` from predicted high/low vs current price.
      - Inserts a `high_confidence_trades` row.
      - Sends a Telegram alert for this trade.

Alerts
------

File: `src/alternator/alerts/telegram_client.py`

- `TelegramClient(bot_token, chat_id)`:
  - `send_message(text)` – sends a plain text message via Telegram API.
  - `send_trade_alert(trade_dict)` – formats and sends trade details:
    - symbol, direction, timeframe, entry, target, stop, expected return, confidence.

Triggered by:
- `backtest_model()` whenever a high-confidence trade is logged.

Interactive CLI
---------------

File: `src/alternator/cli/menu.py`

Entry: `python run_cli.py`

Main menu:
- `1) Download data (yfinance)`
  - Prompts: symbol, start, end, interval.
  - Saves CSV to `data/`.
- `2) Train hierarchical model`
  - Prompts: CSV filename in `data/`.
  - Optional override: epochs.
  - Trains model and saves to `models/hierarchical_model.pth`.
- `3) Backtest model`
  - Prompts: CSV filename in `data/` and model filename in `models/`.
  - Runs backtest, logs predictions/trades, sends Telegram alerts.
- `4) Exit`

Web Server
----------

File: `src/alternator/server/app.py`

- FastAPI application created via `create_app()`.
  - On startup:
    - Loads configuration.
    - Initializes databases.
  - Routes:
    - `GET /` – HTML dashboard (`templates/index.html`).
      - Shows recent predictions and high-confidence trades.
    - `GET /api/predictions` – JSON list of recent predictions.
    - `GET /api/trades` – JSON list of recent trades.

Entry script: `run_server.py`

Running `python run_server.py`:
- Loads config.
- Creates FastAPI app.
- Runs Uvicorn server using `server.host` and `server.port`.

