from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from alternator.config import load_app_config
from alternator.db import (
    init_databases,
    fetch_recent_predictions,
    fetch_recent_high_confidence_trades,
)


def create_app() -> FastAPI:
    app_config = load_app_config()
    init_databases(app_config.paths.db_dir)

    app = FastAPI(title="Alternator Trading Platform")

    static_dir = Path(__file__).resolve().parent / "static"
    templates_dir = Path(__file__).resolve().parent / "templates"
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    templates = Jinja2Templates(directory=str(templates_dir))

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> Any:
        preds = fetch_recent_predictions(app_config.paths.db_dir, limit=50)
        trades = fetch_recent_high_confidence_trades(app_config.paths.db_dir, limit=50)
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "predictions": preds,
                "trades": trades,
            },
        )

    @app.get("/api/predictions")
    async def api_predictions() -> List[Dict[str, Any]]:
        return fetch_recent_predictions(app_config.paths.db_dir, limit=100)

    @app.get("/api/trades")
    async def api_trades() -> List[Dict[str, Any]]:
        return fetch_recent_high_confidence_trades(app_config.paths.db_dir, limit=100)

    return app


