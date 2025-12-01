# AutoTrade2 Backend - FastAPI Dashboard

FastAPI-based frontend for ML trading predictions with backtesting, trade tracking, and alerts.

## Features

### Phase 1 (MVP) - COMPLETED ✅
- **Live Predictions Dashboard** - View latest prediction with confidence scores
- **Prediction History** - Browse past predictions with accuracy tracking
- **Backtest Simulation** - Test strategy on historical data (1,207 predictions with actuals)
- **Clean UI** - HTMX + Tailwind CSS + Plotly.js

### Phase 2 (Planned)
- Trade tracking and P/L management
- Email/Telegram alerts
- Enhanced backtest analytics

### Phase 3 (Planned)
- Full backtesting engine (VectorBT)
- Cloud deployment (Docker + Railway)
- OAuth authentication

## Quick Start

### 1. Install Dependencies

```bash
pip install -r backend/requirements.txt
```

### 2. Initialize Database

```bash
python backend/app/models/database.py
```

This creates 4 new tables in `data/predictions.db`:
- `trades` - Manual trade tracking
- `backtests` - Backtest run results
- `alerts` - Alert configuration
- `users` - User accounts

### 3. Start Server

```bash
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

Or use the startup script:

```bash
./backend/start.sh
```

### 4. Open Dashboard

Navigate to: **http://localhost:8000**

## API Endpoints

### Health Check
```bash
GET /api/health
```

### Predictions
```bash
GET  /api/predictions/latest        # Latest prediction (HTML fragment)
GET  /api/predictions/history       # Recent predictions (HTML table)
POST /api/predictions/{id}/validate # Update with actuals
```

### Backtesting
```bash
POST /api/backtests/simulate        # Run backtest simulation
GET  /api/backtests/{id}            # Get backtest results
```

### Trades (Coming Soon)
```bash
GET  /api/trades                    # List trades
POST /api/trades                    # Log new trade
PUT  /api/trades/{id}               # Update trade
GET  /api/trades/performance        # Performance metrics
```

### Alerts (Coming Soon)
```bash
GET  /api/alerts                    # Alert configs
POST /api/alerts                    # Create alert
GET  /api/alerts/history            # Alert history
```

## Tech Stack

- **Backend**: FastAPI 0.104+
- **ORM**: Peewee 3.17
- **Database**: SQLite (upgrade to PostgreSQL in Phase 2)
- **Frontend**: HTMX + Alpine.js + Tailwind CSS
- **Charts**: Plotly.js
- **Server**: Uvicorn

## Database Schema

### Existing: `predictions` table
- 2,754 total predictions
- 1,207 with actuals (for validation)
- 45+ columns including layer predictions, multi-task outputs

### New tables:

**trades**
```sql
id, prediction_id, entry_time, entry_price,
exit_time, exit_price, pnl, pnl_pct, quantity, notes
```

**backtests**
```sql
id, name, run_date, start_date, end_date,
total_trades, win_rate, total_pnl, sharpe_ratio,
max_drawdown, config_json
```

**alerts**
```sql
id, type, condition_json, triggered_at,
sent_via, prediction_id, dismissed
```

**users**
```sql
id, username, email, password_hash, created_at
```

## Backtest Results (Nov 2025 Data)

Running on 1,207 predictions with actuals:
- **Total Trades**: 125 (confidence > 0.7)
- **Win Rate**: 26.4% (needs improvement - model only trained for 2 epochs)
- **Total P&L**: +6.25% (profitable!)
- **Sharpe Ratio**: 0.40
- **Max Drawdown**: -35.70%

## Development

### Project Structure

```
backend/
├── app/
│   ├── main.py              # FastAPI app entry point
│   ├── models/
│   │   └── database.py      # Peewee ORM models
│   ├── routers/
│   │   ├── predictions.py   # Predictions API
│   │   ├── backtests.py     # Backtesting API
│   │   ├── trades.py        # Trade management API
│   │   └── alerts.py        # Alerts API
│   └── services/
│       ├── prediction_service.py   # ML inference wrapper
│       └── backtest_service.py     # Backtesting engine
├── templates/
│   └── dashboard.html       # Main dashboard UI
├── static/
│   ├── css/
│   └── js/
└── requirements.txt
```

### Adding New Features

1. Create service in `backend/app/services/`
2. Add router in `backend/app/routers/`
3. Include router in `main.py`
4. Update dashboard template

### Running Tests

```bash
pytest backend/tests/  # Coming soon
```

## Deployment

### Local (Current)
```bash
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
```

### Docker (Phase 3)
```bash
docker-compose up -d
```

### Cloud (Phase 3)
Deploy to Railway, Render, or AWS with one click.

## Troubleshooting

### Server won't start
- Check if port 8000 is in use: `lsof -i :8000`
- Ensure dependencies installed: `pip install -r backend/requirements.txt`

### Database errors
- Reinitialize: `python backend/app/models/database.py`
- Check path: `data/predictions.db` must exist

### No predictions showing
- Run training first: `python train_hierarchical.py`
- Or check existing predictions: `sqlite3 data/predictions.db "SELECT COUNT(*) FROM predictions;"`

## Next Steps

- [ ] Add authentication (Phase 2)
- [ ] Implement trade logging
- [ ] Add email/Telegram alerts
- [ ] Full backtesting engine (VectorBT)
- [ ] Docker deployment
- [ ] Cloud hosting

## License

MIT
