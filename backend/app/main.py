"""
FastAPI main application entry point

v2.0: Added WebSocket support and background prediction scheduler
"""
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List
import asyncio
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
from backend.app.services.alert_service import alert_service
import config

logger = logging.getLogger(__name__)

# Rate limiter setup
limiter = Limiter(key_func=get_remote_address)


# =============================================================================
# WebSocket Connection Manager
# =============================================================================
class ConnectionManager:
    """Manages WebSocket connections for real-time updates."""

    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        """Send message to all connected clients."""
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                disconnected.append(connection)

        # Clean up failed connections
        for conn in disconnected:
            self.disconnect(conn)


ws_manager = ConnectionManager()


# =============================================================================
# Background Prediction Scheduler
# =============================================================================
scheduler = AsyncIOScheduler()


async def prediction_loop():
    """
    Background task: Generate predictions every 15 minutes.
    Broadcasts to WebSocket clients and sends Telegram alerts.
    """
    from backend.app.services.prediction_service import prediction_service

    logger.info("Running scheduled prediction...")

    try:
        # Generate fresh prediction
        prediction = await asyncio.to_thread(
            prediction_service.get_prediction,
            force_refresh=True
        )

        if prediction:
            # Broadcast to WebSocket clients
            await ws_manager.broadcast({
                "type": "prediction_update",
                "data": prediction
            })
            logger.info(f"Broadcast prediction to {len(ws_manager.active_connections)} clients")

            # Check and send Telegram alert
            await alert_service.check_and_send(prediction)

    except Exception as e:
        logger.error(f"Prediction loop error: {e}")


# =============================================================================
# Lifespan Manager (startup/shutdown)
# =============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage startup and shutdown events."""
    # Startup
    refresh_minutes = getattr(config, 'PREDICTION_REFRESH_MINUTES', 15)

    scheduler.add_job(
        prediction_loop,
        'interval',
        minutes=refresh_minutes,
        id='prediction_loop',
        replace_existing=True
    )
    scheduler.start()
    logger.info(f"Prediction scheduler started (every {refresh_minutes} min)")

    # Send startup notification
    await alert_service.send_startup_message()

    yield

    # Shutdown
    scheduler.shutdown(wait=False)
    logger.info("Scheduler shut down")


app = FastAPI(
    title="AutoTrade2 API",
    description="ML-powered trading predictions with hierarchical LNN",
    version="2.0.0",
    lifespan=lifespan
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
        "version": "2.0.0",
        "model_loaded": prediction_service.is_model_loaded(),
        "websocket_clients": len(ws_manager.active_connections),
        "scheduler_running": scheduler.running
    }


# =============================================================================
# WebSocket Endpoint for Real-Time Updates
# =============================================================================
@app.websocket("/ws/predictions")
async def websocket_predictions(websocket: WebSocket):
    """
    WebSocket endpoint for real-time prediction updates.

    Clients receive JSON messages:
    - {"type": "prediction_update", "data": {...}}
    - {"type": "ping", "timestamp": ...}
    """
    await ws_manager.connect(websocket)
    try:
        while True:
            # Keep connection alive with periodic ping
            try:
                data = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=30.0
                )
                # Echo back any received messages as pong
                await websocket.send_json({"type": "pong", "received": data})
            except asyncio.TimeoutError:
                # Send ping to keep connection alive
                await websocket.send_json({"type": "ping", "timestamp": str(asyncio.get_event_loop().time())})
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        ws_manager.disconnect(websocket)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
