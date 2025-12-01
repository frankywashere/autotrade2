"""
Predictions API Router
"""
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from typing import Optional
from datetime import datetime
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from backend.app.models.database import Prediction

router = APIRouter()
templates = Jinja2Templates(directory=str(Path(__file__).parent.parent.parent / "templates"))


@router.get("/latest", response_class=HTMLResponse)
async def get_latest_prediction(request: Request):
    """
    Get the most recent prediction

    Returns:
        HTML fragment with prediction card
    """
    try:
        # Get most recent prediction from database
        prediction = (Prediction
                     .select()
                     .order_by(Prediction.timestamp.desc())
                     .first())

        if not prediction:
            return """
            <div id="prediction-card" class="bg-gray-800 rounded-lg p-6">
                <p class="text-yellow-400">No predictions found. Run the model first!</p>
            </div>
            """

        # Calculate target prices
        target_high_price = prediction.current_price * (1 + prediction.predicted_high / 100)
        target_low_price = prediction.current_price * (1 + prediction.predicted_low / 100)

        # Confidence color
        if prediction.confidence >= 0.8:
            conf_color = "text-green-400"
            conf_badge = "🟢 High"
        elif prediction.confidence >= 0.6:
            conf_color = "text-yellow-400"
            conf_badge = "🟡 Medium"
        else:
            conf_color = "text-red-400"
            conf_badge = "🔴 Low"

        return f"""
        <div id="prediction-card" class="bg-gray-800 rounded-lg p-6 border border-gray-700">
            <div class="flex justify-between items-start mb-4">
                <div>
                    <h2 class="text-2xl font-bold text-blue-400">Latest Prediction</h2>
                    <p class="text-sm text-gray-400">{prediction.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
                <div class="text-right">
                    <div class="text-sm text-gray-400">Confidence</div>
                    <div class="text-2xl font-bold {conf_color}">{prediction.confidence:.0%}</div>
                    <div class="text-xs {conf_color}">{conf_badge}</div>
                </div>
            </div>

            <div class="grid grid-cols-3 gap-4 mb-6">
                <div class="bg-gray-700 rounded-lg p-4">
                    <div class="text-sm text-gray-400 mb-1">Current Price</div>
                    <div class="text-2xl font-bold">${prediction.current_price:.2f}</div>
                </div>
                <div class="bg-green-900 bg-opacity-30 rounded-lg p-4 border border-green-700">
                    <div class="text-sm text-green-400 mb-1">Target High</div>
                    <div class="text-2xl font-bold text-green-400">${target_high_price:.2f}</div>
                    <div class="text-xs text-green-400">{prediction.predicted_high:+.2f}%</div>
                </div>
                <div class="bg-red-900 bg-opacity-30 rounded-lg p-4 border border-red-700">
                    <div class="text-sm text-red-400 mb-1">Target Low</div>
                    <div class="text-2xl font-bold text-red-400">${target_low_price:.2f}</div>
                    <div class="text-xs text-red-400">{prediction.predicted_low:+.2f}%</div>
                </div>
            </div>

            {"<div class='mt-4 p-3 bg-green-900 bg-opacity-30 border border-green-700 rounded-lg'>" +
             "<p class='text-sm text-green-400'>✅ Actual prices available - prediction validated</p></div>"
             if prediction.has_actuals else ""}
        </div>
        """

    except Exception as e:
        return f"""
        <div id="prediction-card" class="bg-gray-800 rounded-lg p-6 border border-red-700">
            <p class="text-red-400">Error loading prediction: {str(e)}</p>
        </div>
        """


@router.get("/history", response_class=HTMLResponse)
async def get_prediction_history(
    request: Request,
    limit: int = 10,
    offset: int = 0,
    has_actuals: Optional[bool] = None
):
    """
    Get paginated prediction history

    Args:
        limit: Number of predictions to return
        offset: Pagination offset
        has_actuals: Filter by whether actuals are available

    Returns:
        HTML table with predictions
    """
    try:
        query = Prediction.select().order_by(Prediction.timestamp.desc())

        if has_actuals is not None:
            query = query.where(Prediction.has_actuals == has_actuals)

        predictions = list(query.limit(limit).offset(offset))

        if not predictions:
            return "<div class='text-gray-400'>No predictions found</div>"

        rows = ""
        for pred in predictions:
            accuracy = ""
            if pred.has_actuals:
                # Calculate simple accuracy (did predicted range contain actual?)
                actual_in_range = (
                    pred.actual_high >= pred.current_price * (1 + pred.predicted_low / 100) and
                    pred.actual_low <= pred.current_price * (1 + pred.predicted_high / 100)
                )
                accuracy = "✅" if actual_in_range else "❌"

            rows += f"""
            <tr class="border-b border-gray-700 hover:bg-gray-700">
                <td class="px-4 py-3 text-sm">{pred.timestamp.strftime('%Y-%m-%d %H:%M')}</td>
                <td class="px-4 py-3 text-sm">${pred.current_price:.2f}</td>
                <td class="px-4 py-3 text-sm text-green-400">{pred.predicted_high:+.2f}%</td>
                <td class="px-4 py-3 text-sm text-red-400">{pred.predicted_low:+.2f}%</td>
                <td class="px-4 py-3 text-sm">{pred.confidence:.0%}</td>
                <td class="px-4 py-3 text-sm text-center">{accuracy}</td>
            </tr>
            """

        return f"""
        <div class="bg-gray-800 rounded-lg overflow-hidden">
            <table class="w-full">
                <thead class="bg-gray-700">
                    <tr>
                        <th class="px-4 py-3 text-left text-xs font-medium text-gray-300 uppercase">Time</th>
                        <th class="px-4 py-3 text-left text-xs font-medium text-gray-300 uppercase">Price</th>
                        <th class="px-4 py-3 text-left text-xs font-medium text-gray-300 uppercase">High %</th>
                        <th class="px-4 py-3 text-left text-xs font-medium text-gray-300 uppercase">Low %</th>
                        <th class="px-4 py-3 text-left text-xs font-medium text-gray-300 uppercase">Confidence</th>
                        <th class="px-4 py-3 text-center text-xs font-medium text-gray-300 uppercase">Accuracy</th>
                    </tr>
                </thead>
                <tbody>
                    {rows}
                </tbody>
            </table>
        </div>
        """

    except Exception as e:
        return f"<div class='text-red-400'>Error: {str(e)}</div>"


@router.post("/{prediction_id}/validate")
async def validate_prediction(prediction_id: int):
    """
    Update prediction with actual prices (for performance tracking)

    Args:
        prediction_id: ID of prediction to validate

    Returns:
        Updated prediction with accuracy metrics
    """
    # TODO: Implement validation logic
    raise HTTPException(status_code=501, detail="Not implemented yet")
