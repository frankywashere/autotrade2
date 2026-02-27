V15 System Architecture (v15/)

Scope
- Focused on `v15/` modules only.
- Note: `v15.data` resolves to the package in `v15/data/__init__.py`, not the legacy module `v15/data.py`.

1. DATA LOADING
Implementation
- `v15/data/__init__.py` (exports loader, resampler, native TF helpers)
- `v15/data/loader.py` (CSV loader + validation + alignment)
- `v15/data/resampler.py` (partial-bar resampling utilities)
- `v15/data/native_tf.py` (yfinance native timeframe fetch + cache)
- `v15/data.py` (legacy module, not used by `v15.data` imports)
Status
- Working and wired: `v15.data.load_market_data` is used by `v15/pipeline.py`, `v15/scanner.py`, `v15/dashboard.py`, and inspectors.
- `v15/data.py` appears unused and shadowed by the package (legacy).
Data flow
- Input: CSVs `TSLA_1min.csv`, `SPY_1min.csv`, `VIX_History.csv` in a data directory.
- `load_market_data()` -> parse timestamps -> resample to 5-min bars -> align SPY/VIX to TSLA index (ffill) -> return `(tsla_df, spy_df, vix_df)`.
- Optional path: `native_tf.py` can fetch OHLCV via yfinance and cache to `~/.x14/native_tf_cache`, but not used by the main pipeline.

2. CHANNEL DETECTION (PASS 1)
Implementation
- `v15/labels.py` (`detect_all_channels`, `_detect_tf_window_worker`)
- `v7/core/channel.py` (`detect_channel`)
- `v7/core/timeframe.py` (`resample_ohlc`)
Status
- Working and wired: `scan_channels_two_pass()` in `v15/scanner.py` invokes `detect_all_channels()` for TSLA and SPY.
Data flow
- Input: 5-min DataFrame for a symbol.
- For each TF/window: resample -> slide window by `step` -> run `detect_channel()` -> build `DetectedChannel` objects.
- Output: `ChannelMap` keyed by `(tf, window)` plus `resampled_dfs` per TF.

3. LABEL GENERATION (PASS 2)
Implementation
- `v15/labels.py` (`generate_all_labels`, `label_channel_forward_scan`, `label_channel_from_map`)
- `v15/core/break_scanner.py` (`scan_for_break`)
Status
- Working and wired: `scan_channels_two_pass()` uses `generate_all_labels()` with `labeling_method="forward_scan"` for TSLA and SPY.
- Hybrid labeling (`labeling_method="hybrid"`) is implemented but not used in the scanner.
Data flow
- Input: `ChannelMap` + `resampled_dfs`.
- `forward_scan` path: per channel -> `scan_for_break()` on forward bars (high/low/close) -> compute first break, false breaks, and permanent break -> return `ChannelLabels`.
- `hybrid` path: same scan for break timing, but next channel direction from map lookup.
- Output: `LabeledChannelMap` with `ChannelLabels` containing first break metrics and permanent break tracking.

4. CACHE SYSTEM (SAMPLES)
Implementation
- `v15/scanner.py` (writes pickle samples, partial save on interrupt)
- `v15/pipeline.py` (scan command saves samples to pickle)
- `v15/training/dataset.py` (`load_samples`)
- `v15/dual_inspector.py`, `v15/deprecated_inspector.py` (load caches for inspection)
- `v15/data/native_tf.py` (separate cache for yfinance data)
- `v15/dashboard.py` (Streamlit cache decorators for data/model)
Status
- Working and wired: sample caches are raw `pickle` files containing `List[ChannelSample]`.
- No explicit cache invalidation or versioning; compatibility is best-effort and handled in inspectors.
Data flow
- Input: `List[ChannelSample]` from scanner -> pickle file on disk.
- Output: loaded back into dataset/inspectors via `pickle.load`.
- Optional data cache: yfinance native TF data cached in `~/.x14/native_tf_cache`.

5. FEATURE EXTRACTION (TF EXTRACTOR + FEATURE SYSTEM)
Implementation
- `v15/features/tf_extractor.py` (primary orchestrator, used by scanner/inference)
- `v15/features/*` (tsla_price, technical, spy, vix, cross_asset, tsla_channel, spy_channel, channel_correlation, window_scores, channel_history, events)
- `v15/data/resampler.py` (partial-bar resampling)
- `v7/core/channel.py` (`detect_channels_multi_window`, `select_best_channel`)
- `v15/features/extractor.py` (alternate/legacy orchestrator with optional native TF data)
Status
- Working and wired: `extract_all_tf_features()` is called in `v15/scanner.py` and `v15/inference.py`.
- `v15/features/extractor.py` supports native TF data but is not referenced by scanner/inference (appears unused).
Data flow
- Input: base 5-min `tsla_df`, `spy_df`, `vix_df` plus timestamp.
- For each TF: resample (partial bars), detect channels for TSLA/SPY at 8 windows, extract window-independent features, per-window channel features, cross-asset/channel-correlation, window scores, channel history.
- Add TF-independent event features + bar metadata features.
- Output: flat dict of ~13,660 TF-prefixed features with validated floats.

6. TRAINING PIPELINE (dataset.py, trainer.py)
Implementation
- `v15/training/dataset.py` (ChannelDataset + label extraction)
- `v15/training/trainer.py` (losses, optimization, checkpointing)
- `v15/core/window_strategy.py` (window selection strategies)
- `v15/models/full_model.py`, `v15/models/prediction_heads.py`
- `v15/pipeline.py` (train command)
Status
- Partially wired.
- Core labels (duration, break direction, new channel, permanent_break) are read from `ChannelLabels` and can train the base heads.
- Break-scan and cross-correlation heads are not correctly wired end-to-end:
  - `ChannelLabels` does not define `break_trigger_tf` or `break_return`, but `dataset.py` accesses them directly.
  - `trainer.py` expects label keys like `tsla_bars_to_break`, `tsla_returned`, `cross_direction_aligned`, etc., while `dataset.py` emits `bars_to_first_break`, `returned_to_channel`, `break_direction_aligned`, etc.
  - `PredictionHeads` optional TSLA/SPY/cross heads are disabled by default in `create_model()`, so these losses are effectively dormant unless manually enabled (and then label key mismatches will surface).
Data flow
- Input: `samples.pkl` -> `ChannelDataset` -> `DataLoader`.
- Window selection: strategy chooses `sample.best_window` or heuristics.
- Output: `V15Model` checkpoint with base heads trained; optional heads require manual wiring fixes.

7. MODEL / PREDICTION HEADS
Implementation
- `v15/models/prediction_heads.py`
- `v15/models/full_model.py`
- `v15/models/tf_encoder.py`, `v15/models/cross_tf_attention.py`
Status
- Working for core heads; optional heads are gated by flags.
Targets predicted
- Core:
  - Duration: mean + log_std (Gaussian)
  - Direction: up/down (binary)
  - New channel direction: bear/sideways/bull (3-class)
  - Confidence: calibrated score
- Optional TSLA break-scan heads:
  - Bars to break (Gaussian)
  - Break direction (binary)
  - Break magnitude (Gaussian)
  - Returned to channel (binary)
- Optional SPY break-scan heads:
  - Bars to break (Gaussian)
  - Break direction (binary)
  - Break magnitude (Gaussian)
  - Returned to channel (binary)
- Optional cross-correlation heads:
  - Direction aligned (binary)
  - Who broke first (3-class: TSLA/SPY/simultaneous)
  - Break lag (Gaussian)
  - Both permanent (binary)
  - Return aligned (binary)
- Optional window selector head for learned window selection.

8. INFERENCE
Implementation
- `v15/inference.py` (Predictor)
- `v15/features/tf_extractor.py`
- `v15/models/full_model.py`
- `v7/core/channel.py` (heuristic best window)
Status
- Working and wired.
Data flow
- Input: recent 5-min `tsla_df`, `spy_df`, `vix_df`.
- Feature extraction -> model forward pass -> `Prediction` object.
- Window selection: uses learned window if present, else heuristic `select_best_channel()` on 5-min data.
- Used by `v15/pipeline.py` (infer command) and `v15/dashboard.py`.

9. DASHBOARD / VISUALIZATION
Implementation
- `v15/dashboard.py` (Streamlit app)
- `v15/inference.py` (Predictor)
- `v15/data/__init__.py` (load_market_data)
Status
- Working, but requires a model checkpoint path and data directory.
Data flow
- User selects data dir + model checkpoint -> cached load -> Predictor runs -> metrics and plots rendered.
- Includes feature-importance view if model has explicit weights.

Additional wiring notes
- `v15/pipeline.py` imports `scan_channels` from `v15/scanner.py`, but the scanner only defines `scan_channels_two_pass()`; the scan CLI path appears broken unless an alias is added.
