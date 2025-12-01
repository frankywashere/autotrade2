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
from backend.app.services.prediction_service import prediction_service

router = APIRouter()
templates = Jinja2Templates(directory=str(Path(__file__).parent.parent.parent / "templates"))


@router.get("/latest", response_class=HTMLResponse)
async def get_latest_prediction(request: Request):
    """
    Get the most recent prediction (with 5-min caching)

    Returns:
        HTML fragment with prediction card
    """
    try:
        # Try to generate/get cached prediction
        try:
            pred_dict = prediction_service.get_latest_prediction(force_refresh=False)

            # Convert dict to display format
            prediction_age = (datetime.now() - pred_dict['timestamp']).total_seconds() / 60
            age_text = f"{int(prediction_age)} min ago" if prediction_age < 60 else f"{int(prediction_age/60)} hours ago"

            # Use the prediction dict directly
            current_price = pred_dict['current_price']
            predicted_high = pred_dict['predicted_high']
            predicted_low = pred_dict['predicted_low']
            confidence = pred_dict['confidence']
            timestamp_str = pred_dict['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
            is_cached = prediction_age < 5

        except Exception as e:
            print(f"Failed to generate live prediction: {e}")
            # Fallback to database
            prediction = (Prediction
                         .select()
                         .order_by(Prediction.timestamp.desc())
                         .first())

            if not prediction:
                return """
                <div id="prediction-card" class="bg-gray-800 rounded-lg p-6">
                    <p class="text-yellow-400">No predictions found. Click 'Generate New Prediction' to create one!</p>
                </div>
                """

            # Use database prediction
            current_price = prediction.current_price
            predicted_high = prediction.predicted_high
            predicted_low = prediction.predicted_low
            confidence = prediction.confidence
            timestamp_str = prediction.timestamp.strftime('%Y-%m-%d %H:%M:%S')
            prediction_age = (datetime.now() - prediction.timestamp).total_seconds() / 60
            age_text = f"{int(prediction_age)} min ago" if prediction_age < 60 else f"{int(prediction_age/60)} hours ago"
            is_cached = False

        # Calculate target prices
        target_high_price = current_price * (1 + predicted_high / 100)
        target_low_price = current_price * (1 + predicted_low / 100)

        # Confidence color
        if confidence >= 0.8:
            conf_color = "text-green-400"
            conf_badge = "🟢 High"
        elif confidence >= 0.6:
            conf_color = "text-yellow-400"
            conf_badge = "🟡 Medium"
        else:
            conf_color = "text-red-400"
            conf_badge = "🔴 Low"

        # Cache indicator
        cache_badge = f"<span class='text-xs text-green-400'>✓ Cached ({age_text})</span>" if is_cached else f"<span class='text-xs text-gray-400'>⏰ {age_text}</span>"

        return f"""
        <div id="prediction-card" class="bg-gray-800 rounded-lg p-6 border border-gray-700">
            <div class="flex justify-between items-start mb-4">
                <div>
                    <h2 class="text-2xl font-bold text-blue-400">Latest Prediction</h2>
                    <p class="text-sm text-gray-400">{timestamp_str}</p>
                    <p class="text-xs">{cache_badge}</p>
                </div>
                <div class="text-right">
                    <div class="text-sm text-gray-400">Confidence</div>
                    <div class="text-2xl font-bold {conf_color}">{confidence:.0%}</div>
                    <div class="text-xs {conf_color}">{conf_badge}</div>
                </div>
            </div>

            <div class="grid grid-cols-3 gap-4 mb-6">
                <div class="bg-gray-700 rounded-lg p-4">
                    <div class="text-sm text-gray-400 mb-1">Current Price</div>
                    <div class="text-2xl font-bold">${current_price:.2f}</div>
                </div>
                <div class="bg-green-900 bg-opacity-30 rounded-lg p-4 border border-green-700">
                    <div class="text-sm text-green-400 mb-1">Target High</div>
                    <div class="text-2xl font-bold text-green-400">${target_high_price:.2f}</div>
                    <div class="text-xs text-green-400">{predicted_high:+.2f}%</div>
                </div>
                <div class="bg-red-900 bg-opacity-30 rounded-lg p-4 border border-red-700">
                    <div class="text-sm text-red-400 mb-1">Target Low</div>
                    <div class="text-2xl font-bold text-red-400">${target_low_price:.2f}</div>
                    <div class="text-xs text-red-400">{predicted_low:+.2f}%</div>
                </div>
            </div>
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


@router.post("/generate", response_class=HTMLResponse)
async def generate_new_prediction(request: Request):
    """
    Force generation of new prediction (ignores cache)

    Returns:
        HTML with fresh prediction or error message
    """
    try:
        print("Generating fresh prediction (user requested)...")
        pred_dict = prediction_service.get_latest_prediction(force_refresh=True)

        # Format same as /latest endpoint
        current_price = pred_dict['current_price']
        predicted_high = pred_dict['predicted_high']
        predicted_low = pred_dict['predicted_low']
        confidence = pred_dict['confidence']
        timestamp_str = pred_dict['timestamp'].strftime('%Y-%m-%d %H:%M:%S')

        target_high_price = current_price * (1 + predicted_high / 100)
        target_low_price = current_price * (1 + predicted_low / 100)

        conf_color = "text-green-400" if confidence >= 0.8 else "text-yellow-400" if confidence >= 0.6 else "text-red-400"
        conf_badge = "🟢 High" if confidence >= 0.8 else "🟡 Medium" if confidence >= 0.6 else "🔴 Low"

        return f"""
        <div id="prediction-card" class="bg-gray-800 rounded-lg p-6 border border-green-700">
            <div class="bg-green-900 bg-opacity-20 border border-green-700 rounded-lg p-2 mb-4">
                <p class="text-sm text-green-400">✅ Fresh prediction generated!</p>
            </div>
            <div class="flex justify-between items-start mb-4">
                <div>
                    <h2 class="text-2xl font-bold text-blue-400">Latest Prediction</h2>
                    <p class="text-sm text-gray-400">{timestamp_str}</p>
                    <p class="text-xs text-green-400">✓ Just now</p>
                </div>
                <div class="text-right">
                    <div class="text-sm text-gray-400">Confidence</div>
                    <div class="text-2xl font-bold {conf_color}">{confidence:.0%}</div>
                    <div class="text-xs {conf_color}">{conf_badge}</div>
                </div>
            </div>

            <div class="grid grid-cols-3 gap-4 mb-6">
                <div class="bg-gray-700 rounded-lg p-4">
                    <div class="text-sm text-gray-400 mb-1">Current Price</div>
                    <div class="text-2xl font-bold">${current_price:.2f}</div>
                </div>
                <div class="bg-green-900 bg-opacity-30 rounded-lg p-4 border border-green-700">
                    <div class="text-sm text-green-400 mb-1">Target High</div>
                    <div class="text-2xl font-bold text-green-400">${target_high_price:.2f}</div>
                    <div class="text-xs text-green-400">{predicted_high:+.2f}%</div>
                </div>
                <div class="bg-red-900 bg-opacity-30 rounded-lg p-4 border border-red-700">
                    <div class="text-sm text-red-400 mb-1">Target Low</div>
                    <div class="text-2xl font-bold text-red-400">${target_low_price:.2f}</div>
                    <div class="text-xs text-red-400">{predicted_low:+.2f}%</div>
                </div>
            </div>
        </div>
        """

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return f"""
        <div id="prediction-card" class="bg-red-900 bg-opacity-30 border border-red-700 rounded-lg p-6">
            <p class="text-red-400 font-bold">❌ Failed to generate prediction</p>
            <p class="text-sm text-gray-400 mt-2">Error: {str(e)}</p>
            <details class="mt-4">
                <summary class="text-xs text-gray-500 cursor-pointer">Show details</summary>
                <pre class="text-xs text-gray-500 mt-2 overflow-auto">{error_details}</pre>
            </details>
        </div>
        """


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
