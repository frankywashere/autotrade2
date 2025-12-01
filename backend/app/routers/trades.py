"""
Trade Management API Router
"""
from fastapi import APIRouter, HTTPException, Request, Form
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from datetime import datetime
from typing import Optional
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from backend.app.models.database import Trade, Prediction
from backend.app.services.performance_service import performance_service

router = APIRouter()


class TradeCreate(BaseModel):
    """Trade entry request"""
    prediction_id: Optional[int] = None
    entry_time: datetime
    entry_price: float
    quantity: int = 1
    notes: Optional[str] = None


class TradeUpdate(BaseModel):
    """Trade exit request"""
    exit_time: datetime
    exit_price: float
    notes: Optional[str] = None


@router.get("/", response_class=HTMLResponse)
async def get_trades(request: Request, limit: int = 50, offset: int = 0):
    """
    Get list of manual trades

    Returns:
        HTML table with trades
    """
    try:
        trades = (Trade
                 .select()
                 .order_by(Trade.entry_time.desc())
                 .limit(limit)
                 .offset(offset))

        trade_list = list(trades)

        if not trade_list:
            return """
            <div class="bg-gray-800 rounded-lg p-6">
                <p class="text-gray-400">No trades logged yet. Start by logging your first trade below!</p>
            </div>
            """

        rows = ""
        for trade in trade_list:
            # Status and colors
            if trade.exit_time:
                status = "Closed"
                status_color = "bg-gray-700"
                pnl_color = "text-green-400" if trade.pnl and trade.pnl > 0 else "text-red-400"
            else:
                status = "Open"
                status_color = "bg-blue-900"
                pnl_color = "text-gray-400"

            rows += f"""
            <tr class="border-b border-gray-700 hover:bg-gray-700">
                <td class="px-4 py-3 text-sm">{trade.entry_time.strftime('%Y-%m-%d %H:%M')}</td>
                <td class="px-4 py-3 text-sm">${trade.entry_price:.2f}</td>
                <td class="px-4 py-3 text-sm">${trade.exit_price:.2f if trade.exit_price else '-'}</td>
                <td class="px-4 py-3 text-sm {pnl_color} font-bold">
                    {f'${trade.pnl:.2f}' if trade.pnl else '-'}
                    {f' ({trade.pnl_pct:+.2f}%)' if trade.pnl_pct else ''}
                </td>
                <td class="px-4 py-3 text-sm">{trade.quantity}</td>
                <td class="px-4 py-3">
                    <span class="px-2 py-1 text-xs rounded {status_color}">{status}</span>
                </td>
                <td class="px-4 py-3 text-sm">
                    {'<button class="text-blue-400 hover:text-blue-300" onclick="editTrade(' + str(trade.id) + ')">Edit</button>' if not trade.exit_time else ''}
                </td>
            </tr>
            """

        return f"""
        <div class="bg-gray-800 rounded-lg overflow-hidden">
            <table class="w-full">
                <thead class="bg-gray-700">
                    <tr>
                        <th class="px-4 py-3 text-left text-xs font-medium text-gray-300 uppercase">Entry Time</th>
                        <th class="px-4 py-3 text-left text-xs font-medium text-gray-300 uppercase">Entry Price</th>
                        <th class="px-4 py-3 text-left text-xs font-medium text-gray-300 uppercase">Exit Price</th>
                        <th class="px-4 py-3 text-left text-xs font-medium text-gray-300 uppercase">P&L</th>
                        <th class="px-4 py-3 text-left text-xs font-medium text-gray-300 uppercase">Quantity</th>
                        <th class="px-4 py-3 text-left text-xs font-medium text-gray-300 uppercase">Status</th>
                        <th class="px-4 py-3 text-left text-xs font-medium text-gray-300 uppercase">Actions</th>
                    </tr>
                </thead>
                <tbody>
                    {rows}
                </tbody>
            </table>
        </div>
        """

    except Exception as e:
        return f"<div class='text-red-400'>Error loading trades: {str(e)}</div>"


@router.post("/", response_class=HTMLResponse)
async def create_trade(
    entry_time: str = Form(...),
    entry_price: float = Form(...),
    quantity: int = Form(1),
    notes: Optional[str] = Form(None)
):
    """
    Log a new manual trade entry

    Returns:
        Success message HTML
    """
    try:
        # Parse entry time
        entry_dt = datetime.fromisoformat(entry_time)

        # Create trade
        trade = Trade.create(
            entry_time=entry_dt,
            entry_price=entry_price,
            quantity=quantity,
            notes=notes
        )

        return f"""
        <div class="bg-green-900 bg-opacity-30 border border-green-700 rounded-lg p-4 mb-4">
            <p class="text-green-400">✅ Trade #{trade.id} logged successfully!</p>
            <p class="text-sm text-gray-400">Entry: ${entry_price:.2f} at {entry_dt.strftime('%Y-%m-%d %H:%M')}</p>
        </div>
        """

    except Exception as e:
        return f"""
        <div class="bg-red-900 bg-opacity-30 border border-red-700 rounded-lg p-4 mb-4">
            <p class="text-red-400">❌ Error logging trade: {str(e)}</p>
        </div>
        """


@router.put("/{trade_id}", response_class=HTMLResponse)
async def update_trade(
    trade_id: int,
    exit_time: str = Form(...),
    exit_price: float = Form(...),
    notes: Optional[str] = Form(None)
):
    """
    Update trade with exit information

    Returns:
        Success message with P&L
    """
    try:
        trade = Trade.get_by_id(trade_id)

        # Parse exit time
        exit_dt = datetime.fromisoformat(exit_time)

        # Calculate P&L
        pnl = (exit_price - trade.entry_price) * trade.quantity
        pnl_pct = ((exit_price / trade.entry_price) - 1) * 100

        # Update trade
        trade.exit_time = exit_dt
        trade.exit_price = exit_price
        trade.pnl = pnl
        trade.pnl_pct = pnl_pct
        if notes:
            trade.notes = notes
        trade.save()

        pnl_color = "text-green-400" if pnl > 0 else "text-red-400"

        return f"""
        <div class="bg-green-900 bg-opacity-30 border border-green-700 rounded-lg p-4 mb-4">
            <p class="text-green-400">✅ Trade #{trade_id} closed!</p>
            <p class="text-sm {pnl_color} font-bold">P&L: ${pnl:.2f} ({pnl_pct:+.2f}%)</p>
        </div>
        """

    except Trade.DoesNotExist:
        return """
        <div class="bg-red-900 bg-opacity-30 border border-red-700 rounded-lg p-4 mb-4">
            <p class="text-red-400">❌ Trade not found</p>
        </div>
        """
    except Exception as e:
        return f"""
        <div class="bg-red-900 bg-opacity-30 border border-red-700 rounded-lg p-4 mb-4">
            <p class="text-red-400">❌ Error: {str(e)}</p>
        </div>
        """


@router.get("/performance", response_class=HTMLResponse)
async def get_performance_summary(request: Request):
    """
    Get overall trading performance metrics

    Returns:
        HTML with performance cards
    """
    try:
        metrics = performance_service.calculate_performance()

        win_rate_color = "text-green-400" if metrics['win_rate'] > 0.5 else "text-yellow-400" if metrics['win_rate'] > 0.3 else "text-red-400"
        pnl_color = "text-green-400" if metrics['total_pnl'] > 0 else "text-red-400"
        sharpe_color = "text-green-400" if metrics['sharpe_ratio'] > 1 else "text-yellow-400" if metrics['sharpe_ratio'] > 0 else "text-red-400"

        return f"""
        <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div class="bg-gray-800 rounded-lg p-4 border border-gray-700">
                <div class="text-sm text-gray-400 mb-1">Total Trades</div>
                <div class="text-2xl font-bold">{metrics['total_trades']}</div>
                <div class="text-xs text-gray-400">{metrics['winning_trades']}W / {metrics['losing_trades']}L</div>
            </div>
            <div class="bg-gray-800 rounded-lg p-4 border border-gray-700">
                <div class="text-sm text-gray-400 mb-1">Win Rate</div>
                <div class="text-2xl font-bold {win_rate_color}">{metrics['win_rate']:.1%}</div>
            </div>
            <div class="bg-gray-800 rounded-lg p-4 border border-gray-700">
                <div class="text-sm text-gray-400 mb-1">Total P&L</div>
                <div class="text-2xl font-bold {pnl_color}">${metrics['total_pnl']:.2f}</div>
                <div class="text-xs {pnl_color}">{metrics['total_pnl_pct']:+.2f}%</div>
            </div>
            <div class="bg-gray-800 rounded-lg p-4 border border-gray-700">
                <div class="text-sm text-gray-400 mb-1">Sharpe Ratio</div>
                <div class="text-2xl font-bold {sharpe_color}">{metrics['sharpe_ratio']:.2f}</div>
            </div>
        </div>

        <div class="grid grid-cols-2 gap-4 mt-4">
            <div class="bg-gray-800 rounded-lg p-4 border border-gray-700">
                <div class="text-sm text-gray-400 mb-1">Avg Win</div>
                <div class="text-xl font-bold text-green-400">${metrics['avg_win']:.2f}</div>
            </div>
            <div class="bg-gray-800 rounded-lg p-4 border border-gray-700">
                <div class="text-sm text-gray-400 mb-1">Avg Loss</div>
                <div class="text-xl font-bold text-red-400">${metrics['avg_loss']:.2f}</div>
            </div>
        </div>

        <div class="mt-4 bg-gray-800 rounded-lg p-4 border border-gray-700">
            <div class="text-sm text-gray-400 mb-1">Max Drawdown</div>
            <div class="text-xl font-bold text-red-400">{metrics['max_drawdown']:.2f}%</div>
        </div>
        """

    except Exception as e:
        return f"<div class='text-red-400'>Error: {str(e)}</div>"
