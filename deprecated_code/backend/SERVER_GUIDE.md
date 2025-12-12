# AutoTrade2 Backend - Server Management Guide

## Quick Start

### Start Server
```bash
cd /Users/frank/Desktop/CodingProjects/autotrade2
python3 -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

Then open in browser: **http://localhost:8000**

---

## Common Issues

### "Address already in use" Error

**Problem:** Port 8000 is already occupied

**Solution 1 - Kill existing server:**
```bash
# Find process using port 8000
lsof -ti:8000

# Kill it
kill $(lsof -ti:8000)

# Now restart
python3 -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

**Solution 2 - Use different port:**
```bash
python3 -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8001 --reload
# Then open: http://localhost:8001
```

---

## Server Commands

### Start (Development Mode with Auto-Reload)
```bash
python3 -m uvicorn backend.app.main:app --reload
```

### Start (Production Mode - No Auto-Reload)
```bash
python3 -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
```

### Start in Background
```bash
nohup python3 -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 > server.log 2>&1 &
```

### Stop Background Server
```bash
kill $(lsof -ti:8000)
```

### Check Server Status
```bash
curl http://localhost:8000/api/health
```

Expected response:
```json
{"status":"healthy","version":"0.1.0","model_loaded":false}
```

---

## Endpoints to Test

### Health Check
```bash
curl http://localhost:8000/api/health
```

### Latest Prediction (HTML)
```bash
curl http://localhost:8000/api/predictions/latest
```

### Run Backtest
```bash
curl -X POST "http://localhost:8000/api/backtests/simulate?start_date=2025-11-01&end_date=2025-11-30&confidence_threshold=0.7"
```

### View API Docs
Open browser: **http://localhost:8000/docs**

---

## Logs

### View Real-Time Logs
```bash
# If running in foreground: logs show in terminal

# If running in background (nohup):
tail -f server.log
```

### Check for Errors
```bash
grep ERROR server.log
```

---

## Virtual Environment

### Using Project Virtual Environment
```bash
# Activate
source myenv/bin/activate

# Start server
python -m uvicorn backend.app.main:app --reload

# Deactivate when done
deactivate
```

---

## Troubleshooting

### Server Won't Start
1. Check Python version: `python3 --version` (need 3.11+)
2. Install dependencies: `pip install -r backend/requirements.txt`
3. Check port: `lsof -i:8000`
4. Check database exists: `ls data/predictions.db`

### Database Errors
```bash
# Reinitialize database
python backend/app/models/database.py
```

### Import Errors
```bash
# Make sure you're in project root
cd /Users/frank/Desktop/CodingProjects/autotrade2

# Check PYTHONPATH
echo $PYTHONPATH

# Start server from project root
python3 -m uvicorn backend.app.main:app --reload
```

### Model Not Loading
```bash
# Check if model file exists
ls -lh models/hierarchical_lnn.pth

# Should see: 95MB file
```

---

## Development Workflow

**1. Make code changes**
```bash
# Edit files in backend/app/
vim backend/app/routers/predictions.py
```

**2. Server auto-reloads (if using --reload flag)**
```
INFO: Detected change in 'backend/app/routers/predictions.py'
INFO: Reloading...
```

**3. Test in browser**
```
http://localhost:8000
```

**4. Check logs for errors**
- Logs show in terminal if running in foreground

---

## Quick Commands Reference

```bash
# START SERVER
python3 -m uvicorn backend.app.main:app --reload

# STOP SERVER (Ctrl+C in terminal, or)
kill $(lsof -ti:8000)

# CHECK STATUS
curl localhost:8000/api/health

# VIEW API DOCS
open http://localhost:8000/docs

# TAIL LOGS
tail -f server.log
```

---

## Next Steps

Now that you know how to manage the server, you can:

1. Start server: `python3 -m uvicorn backend.app.main:app --reload`
2. Open dashboard: `http://localhost:8000`
3. Test features:
   - Live predictions tab
   - Run a backtest
   - Log a trade
   - View performance metrics
