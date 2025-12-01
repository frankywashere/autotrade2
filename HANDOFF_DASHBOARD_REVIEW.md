# Dashboard Code Review - Handoff Document

## Context

I've built a new FastAPI + HTMX trading dashboard to replace the old Streamlit dashboard. The system uses a Hierarchical Liquid Neural Network (LNN) trained on 14,487 features to predict TSLA stock price movements.

## What to Review

Please review the new dashboard implementation for:
- Architecture issues
- Security vulnerabilities
- Performance problems
- Code quality issues
- Potential bugs
- Best practice violations

## System Overview

**Purpose:** ML-powered trading dashboard with live predictions, backtesting, trade tracking, and performance analytics

**Tech Stack:**
- Backend: FastAPI 0.104+ (Python 3.12)
- Frontend: HTMX + Alpine.js + Tailwind CSS
- Database: SQLite (Peewee ORM)
- ML Model: Hierarchical LNN (PyTorch, 14,487 features)
- Charts: Plotly.js

**Key Directories:**
```
backend/
├── app/
│   ├── main.py              # FastAPI entry point
│   ├── models/
│   │   └── database.py      # Peewee ORM models
│   ├── routers/
│   │   ├── predictions.py   # Predictions API
│   │   ├── backtests.py     # Backtesting API
│   │   ├── trades.py        # Trade management API
│   │   ├── alerts.py        # Alerts API
│   │   └── charts.py        # Plotly chart data API
│   └── services/
│       ├── prediction_service.py    # ML inference wrapper
│       ├── backtest_service.py      # Backtesting engine
│       ├── performance_service.py   # Performance analytics
│       ├── online_learner.py        # Online learning (LNN adaptation)
│       └── validation_checker.py    # Hourly validation service
├── templates/
│   └── dashboard.html       # Main UI (HTMX + Tailwind)
└── static/
```

## Core Files to Review (Priority Order)

### 1. Main Application
**File:** `backend/app/main.py`
- FastAPI app setup
- CORS configuration
- Router integration
- Static file serving

**Questions:**
- Is CORS configured securely? (Currently allows all origins)
- Are there any missing error handlers?
- Should we add rate limiting?

### 2. Prediction Service
**File:** `backend/app/services/prediction_service.py` (~230 lines)

**What it does:**
- Loads Hierarchical LNN model (singleton pattern)
- Fetches live data from yfinance (TSLA + SPY)
- Extracts 14,487 features (channels, RSI, correlations, etc.)
- Runs model inference
- Caches predictions for 5 minutes
- Saves predictions to database

**Critical concerns:**
- Model loading happens on first prediction (30+ second delay)
- Feature extraction takes 5-7 minutes on first run per day
- Error handling for yfinance failures
- Memory usage (model is 175MB, feature extraction can use 2-3GB)

**Key method:** `get_latest_prediction(force_refresh: bool)`

### 3. Database Models
**File:** `backend/app/models/database.py` (~180 lines)

**Schema:**
- `predictions` table (existing, 2,754 records)
  - predicted_high, predicted_low, confidence
  - actual_high, actual_low, has_actuals
  - Layer predictions (15min, 1hour, 4hour, daily)
  - Multi-task outputs

- `trades` table (new - manual trade tracking)
- `backtests` table (new - backtest results)
- `alerts` table (new - alert configs)
- `users` table (new - authentication, not implemented yet)

**Questions:**
- Are foreign keys properly indexed?
- Should we add created_at/updated_at timestamps?
- Is SQLite sufficient or should we migrate to PostgreSQL?

### 4. API Routers

**File:** `backend/app/routers/predictions.py` (~300 lines)

**Endpoints:**
- `GET /api/predictions/latest` - Shows cached or generates new prediction
- `POST /api/predictions/generate` - Forces fresh prediction (30+ sec)
- `GET /api/predictions/history` - Recent predictions table

**Concerns:**
- No authentication (anyone can generate predictions)
- No rate limiting (could spam model inference)
- Error messages might leak system info

**File:** `backend/app/routers/backtests.py` (~120 lines)

**Endpoints:**
- `POST /api/backtests/simulate` - Run backtest on historical predictions

**How it works:**
- Uses existing predictions with actuals (1,207 records from Nov 2025)
- Simulates trading strategy with confidence threshold
- Calculates: win rate, P&L, Sharpe ratio, max drawdown

**Concerns:**
- No input validation on date ranges
- Could run expensive backtests concurrently
- Results not paginated (could return large HTML)

**File:** `backend/app/routers/trades.py` (~270 lines)

**Endpoints:**
- `POST /api/trades` - Log trade entry
- `PUT /api/trades/{id}` - Close trade (calculates P&L)
- `GET /api/trades/performance` - Performance metrics

**Concerns:**
- No authentication (anyone can log fake trades)
- No validation on prices (could enter negative/invalid values)
- P&L calculation is simple (no commissions, slippage)

### 5. Frontend Dashboard
**File:** `backend/templates/dashboard.html` (~320 lines)

**Features:**
- 4 tabs: Live Predictions, Backtesting, Trades, Performance
- HTMX for seamless updates (no page reload)
- Alpine.js for tab navigation
- Plotly.js charts (cumulative P&L, returns distribution)
- Auto-refresh (predictions every 5min, performance every 30sec)

**Concerns:**
- No CSRF protection
- No XSS sanitization on user inputs
- All endpoints return raw HTML (not JSON)
- No loading states for slow operations
- Charts load on every tab switch (inefficient)

### 6. Online Learning (Not Yet Integrated)
**File:** `backend/app/services/online_learner.py` (~260 lines)

**What it should do:**
- Monitor prediction errors
- Trigger model.update_online() when errors exceed threshold
- Adapt fusion weights based on layer accuracy
- Daily caps to prevent catastrophic forgetting

**Status:** Implemented but NOT wired to production
- No automatic validation checking
- No error feedback loop
- Model doesn't actually adapt yet

**File:** `backend/app/services/validation_checker.py` (~170 lines)
- Background service to fetch actuals and trigger updates
- NOT started in main.py yet

## Current Issues to Flag

### 1. Performance
- First prediction takes 5-7 minutes (feature extraction)
- Model loads on first request (30 sec delay)
- No connection pooling
- No async I/O for yfinance calls

### 2. Security
- No authentication system
- CORS allows all origins
- No rate limiting
- No input validation on forms
- No CSRF tokens
- Error messages might leak paths

### 3. Reliability
- Single point of failure (if model loading fails, entire app fails)
- No graceful degradation
- SQLite database not suitable for concurrent writes
- No health checks
- No monitoring/alerting

### 4. Code Quality
- Mixture of sync and async (FastAPI endpoints are async but services are sync)
- No type hints in some places
- Hardcoded values (cache TTL, thresholds)
- No unit tests
- No integration tests

### 5. Data Issues
- Feature extraction returns tuple but some code expects DataFrame
- Timezone handling (fixed, but worth double-checking)
- Model dimension mismatch if features change (partially fixed)
- Event data ends Dec 19, 2025 (needs updating)

## Testing the Dashboard

### Setup
```bash
cd /Users/frank/Desktop/CodingProjects/autotrade2
source myenv/bin/activate
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

Open: http://localhost:8000

### Test Cases

**1. Live Predictions Tab**
- Click "Generate New Prediction" button
- Should take 5-10 seconds (features cached) or 5-7 minutes (first run)
- Check if prediction appears with confidence score
- Verify target high/low prices calculated correctly

**2. Backtesting Tab**
- Enter dates: 2025-11-01 to 2025-11-30
- Set confidence: 0.7 (70%)
- Click "Run Backtest"
- Should show results: ~125 trades, ~26% win rate, +6.25% P&L

**3. Trades Tab**
- Log a manual trade (entry time, price, quantity)
- Should save and appear in table
- Update trade with exit price
- Should calculate P&L automatically

**4. Performance Tab**
- Should show metrics cards (win rate, total P&L, Sharpe ratio)
- Charts should load (cumulative P&L, returns distribution)
- If no trades logged, should show "No trades yet"

## Known Bugs/TODOs

1. **Live predictions returns only 2 rows instead of 2,394** (currently debugging)
   - Features extract correctly (13,185 features, 2,394 bars)
   - But only 2 bars make it to model input
   - Likely alignment issue between channel and non-channel features

2. **Model dimension mismatch** (partially fixed)
   - Trained model expects 14,487 features
   - Load function infers input_size from weights
   - Works now, but fragile if features change

3. **Online learning not active**
   - Code exists but not integrated
   - No background validation checker running
   - Model doesn't adapt after deployment yet

4. **No authentication**
   - Anyone can access dashboard
   - Anyone can generate predictions (expensive operation)
   - Anyone can log trades

5. **Background service not implemented**
   - No auto-generation every 15 minutes
   - User must manually click button

## Questions for Review

1. **Architecture:** Is FastAPI + HTMX + SQLite the right stack? Or should we use React + PostgreSQL?

2. **Security:** What's the minimum viable auth? Session-based? JWT? OAuth?

3. **Performance:** Should we preload model on startup? Background worker for predictions?

4. **Data integrity:** How to handle feature dimension changes when retraining?

5. **Testing:** What's the priority: unit tests, integration tests, or E2E tests?

6. **Deployment:** Should we Dockerize now or wait for more features?

7. **Online learning:** Is the current approach (daily caps, layer-specific LR) sound?

8. **Error handling:** Are we gracefully handling yfinance failures, model errors, database errors?

## Files Modified in This Session

**New files (28 total):**
- `backend/` entire directory (12 Python files, 1 HTML template)
- `backend/SERVER_GUIDE.md`
- `backend/README.md`
- Various service and router files

**Modified files:**
- `src/ml/hierarchical_model.py` (fixed load function to infer input_size)
- `train_hierarchical.py` (now saves input_size in checkpoint)
- `deprecated/live_data_feed.py` (fixed timezone bug)
- `models/` (cleaned up old model files, moved to deprecated/)

**Moved to deprecated:**
- `hierarchical_dashboard.py` (old Streamlit dashboard)
- Old LNN models (lnn_15min.pth, lnn_1hour.pth, etc.)

## Git Branch

Current branch: `hierarchical-dataset-updates`
Latest commit: cc8756c "Move old Streamlit dashboard to deprecated folder"

## Additional Context

**Model Training:**
- 11 epochs, val_loss: 0.1819
- Trained on 2015-2025 data
- 14,487 input features (14,322 channel + 165 non-channel)
- Hierarchical architecture: fast/medium/slow layers
- Multi-task learning: 16 prediction heads

**Current Performance (Backtest on Nov 2025):**
- 125 trades (confidence > 0.7)
- Win rate: 26.4% (low - model needs more training)
- Total P&L: +6.25% (profitable despite low win rate!)
- Sharpe ratio: 0.40

**Data:**
- 2,754 predictions in database (1,207 with actuals)
- Features stored in /Volumes/NVME2/featureslabels (90GB)
- Live data from yfinance (TSLA + SPY)

## Review Goals

Please identify:
1. **Critical bugs** that would break production
2. **Security vulnerabilities** (especially pre-auth issues)
3. **Performance bottlenecks** (beyond the known 5-7 min feature extraction)
4. **Architecture improvements** (is this the right approach?)
5. **Code smells** (technical debt, anti-patterns)

Focus on actionable feedback that can be addressed in the next 1-2 days.

Thank you!
