"""
FastAPI main application entry point
"""
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from pathlib import Path
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from backend.app.routers import predictions, backtests, trades, alerts, charts

# Rate limiter setup
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="AutoTrade2 API",
    description="ML-powered trading predictions with hierarchical LNN",
    version="0.1.0"
)

# Add rate limit error handler
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS configuration - restrict to localhost for security
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_credentials=False,  # Not using cross-origin cookies
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Content-Type", "Authorization"],
)

# Static files and templates
static_path = Path(__file__).parent.parent / "static"
templates_path = Path(__file__).parent.parent / "templates"

app.mount("/static", StaticFiles(directory=str(static_path)), name="static")
templates = Jinja2Templates(directory=str(templates_path), autoescape=True)

# Include routers
app.include_router(predictions.router, prefix="/api/predictions", tags=["predictions"])
app.include_router(backtests.router, prefix="/api/backtests", tags=["backtests"])
app.include_router(trades.router, prefix="/api/trades", tags=["trades"])
app.include_router(alerts.router, prefix="/api/alerts", tags=["alerts"])
app.include_router(charts.router, prefix="/api/charts", tags=["charts"])


@app.get("/")
async def root(request: Request):
    """Main dashboard page"""
    return templates.TemplateResponse("dashboard.html", {"request": request})


@app.get("/api/health")
async def health():
    """Health check endpoint"""
    from backend.app.services.prediction_service import prediction_service
    return {
        "status": "healthy",
        "version": "0.1.0",
        "model_loaded": prediction_service.is_model_loaded()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
