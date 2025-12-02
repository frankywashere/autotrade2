"""
Backtesting API Router
"""
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, field_validator
from slowapi import Limiter
from slowapi.util import get_remote_address
from datetime import date
from typing import Optional
import asyncio
import logging

logger = logging.getLogger(__name__)
limiter = Limiter(key_func=get_remote_address)
router = APIRouter()


class BacktestRequest(BaseModel):
    """Backtest configuration with validation"""
    name: str = Field(..., min_length=1, max_length=100)
    start_date: date
    end_date: date
    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    layer: Optional[str] = Field(default=None, pattern="^(fast|medium|slow|fusion)$")

    @field_validator('end_date')
    @classmethod
    def end_after_start(cls, v, info):
        if 'start_date' in info.data and v <= info.data['start_date']:
            raise ValueError('end_date must be after start_date')
        return v

    @field_validator('start_date')
    @classmethod
    def date_not_future(cls, v):
        if v > date.today():
            raise ValueError('start_date cannot be in future')
        return v


@router.post("/simulate")
@limiter.limit("5/minute")
async def run_backtest_simulation(
    request: Request,
    start_date: date,
    end_date: date,
    confidence_threshold: float = 0.7,
    layer: Optional[str] = None
):
    """
    Quick backtest simulation using existing predictions
    Rate limited: 5 requests per minute

    Args:
        request: FastAPI request (for rate limiting)
        start_date: Backtest start date
        end_date: Backtest end date
        confidence_threshold: Minimum confidence to trade
        layer: Which layer to use (15min, 1hour, 4hour, daily, or None for fusion)

    Returns:
        HTML with backtest results
    """
    from backend.app.services.backtest_service import backtest_service

    try:
        # Run blocking operation in thread pool
        results = await asyncio.to_thread(
            backtest_service.run_simulation,
            f"Backtest {start_date} to {end_date}",
            start_date,
            end_date,
            confidence_threshold,
            layer
        )

        # Color coding
        win_rate_color = "text-green-400" if results['win_rate'] > 0.5 else "text-yellow-400" if results['win_rate'] > 0.3 else "text-red-400"
        pnl_color = "text-green-400" if results['total_pnl'] > 0 else "text-red-400"
        sharpe_color = "text-green-400" if results['sharpe_ratio'] > 1 else "text-yellow-400" if results['sharpe_ratio'] > 0 else "text-red-400"

        return f"""
        <div class="bg-gray-700 rounded-lg p-6 border border-gray-600">
            <h3 class="text-xl font-semibold mb-4 text-blue-400">Backtest Results</h3>

            <div class="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                <div class="bg-gray-800 rounded-lg p-4">
                    <div class="text-sm text-gray-400 mb-1">Total Trades</div>
                    <div class="text-2xl font-bold">{results['total_trades']}</div>
                </div>
                <div class="bg-gray-800 rounded-lg p-4">
                    <div class="text-sm text-gray-400 mb-1">Win Rate</div>
                    <div class="text-2xl font-bold {win_rate_color}">{results['win_rate']:.1%}</div>
                </div>
                <div class="bg-gray-800 rounded-lg p-4">
                    <div class="text-sm text-gray-400 mb-1">Total P&L</div>
                    <div class="text-2xl font-bold {pnl_color}">{results['total_pnl']:+.2f}%</div>
                </div>
                <div class="bg-gray-800 rounded-lg p-4">
                    <div class="text-sm text-gray-400 mb-1">Sharpe Ratio</div>
                    <div class="text-2xl font-bold {sharpe_color}">{results['sharpe_ratio']:.2f}</div>
                </div>
            </div>

            <div class="grid grid-cols-2 gap-4 mb-6">
                <div class="bg-gray-800 rounded-lg p-4">
                    <div class="text-sm text-gray-400 mb-1">Avg P&L per Trade</div>
                    <div class="text-xl font-bold">{results['avg_pnl']:+.2f}%</div>
                </div>
                <div class="bg-gray-800 rounded-lg p-4">
                    <div class="text-sm text-gray-400 mb-1">Max Drawdown</div>
                    <div class="text-xl font-bold text-red-400">{results['max_drawdown']:.2f}%</div>
                </div>
            </div>

            <div class="text-sm text-gray-400">
                <p>Period: {start_date} to {end_date}</p>
                <p>Confidence threshold: {confidence_threshold:.0%}</p>
                <p>Layer: {layer or 'Fusion (default)'}</p>
            </div>
        </div>
        """

    except Exception as e:
        logger.exception("Error running backtest")
        return """
        <div class="bg-red-900 bg-opacity-30 border border-red-700 rounded-lg p-4">
            <p class="text-red-400">Error running backtest. Please check your date range and try again.</p>
        </div>
        """


@router.get("/{backtest_id}")
async def get_backtest_results(backtest_id: int):
    """
    Get detailed backtest results

    Args:
        backtest_id: ID of backtest run

    Returns:
        Full backtest results including trade-by-trade breakdown
    """
    # TODO: Implement database query
    raise HTTPException(status_code=404, detail="Backtest not found")
