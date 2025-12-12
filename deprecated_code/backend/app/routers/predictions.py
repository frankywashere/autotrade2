"""
Predictions API Router
"""
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from slowapi import Limiter
from slowapi.util import get_remote_address
from typing import Optional
from datetime import datetime
from pathlib import Path
from html import escape
import asyncio
import logging
import sys

# Rate limiter (shared with main app)
limiter = Limiter(key_func=get_remote_address)

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from backend.app.models.database import Prediction
from backend.app.services.prediction_service import prediction_service

router = APIRouter()
templates = Jinja2Templates(directory=str(Path(__file__).parent.parent.parent / "templates"))
logger = logging.getLogger(__name__)


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
            # Run blocking operation in thread pool to avoid blocking event loop
            pred_dict = await asyncio.to_thread(
                prediction_service.get_latest_prediction, force_refresh=False
            )

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
            logger.warning(f"Failed to generate live prediction: {e}")
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
        logger.exception("Error loading prediction")
        return f"""
        <div id="prediction-card" class="bg-gray-800 rounded-lg p-6 border border-red-700">
            <p class="text-red-400">Error loading prediction. Please try again.</p>
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
        logger.exception("Error loading prediction history")
        return "<div class='text-red-400'>Error loading history. Please try again.</div>"


@router.post("/generate", response_class=HTMLResponse)
@limiter.limit("1/minute")
async def generate_new_prediction(request: Request):
    """
    Force generation of new prediction (ignores cache)
    Rate limited: 1 request per minute

    Returns:
        HTML with fresh prediction or error message
    """
    try:
        logger.info("Generating fresh prediction (user requested)...")
        # Run blocking operation in thread pool to avoid blocking event loop
        pred_dict = await asyncio.to_thread(
            prediction_service.get_latest_prediction, force_refresh=True
        )

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
        logger.exception("Failed to generate prediction")
        return """
        <div id="prediction-card" class="bg-red-900 bg-opacity-30 border border-red-700 rounded-lg p-6">
            <p class="text-red-400 font-bold">❌ Failed to generate prediction</p>
            <p class="text-sm text-gray-400 mt-2">Unable to generate prediction. Please try again later.</p>
        </div>
        """


@router.get("/channel", response_class=HTMLResponse)
async def get_channel_projection(request: Request):
    """
    Get dynamic horizon prediction with confidence-based selection.

    Uses project_channel() to find the shortest horizon with sufficient confidence.
    Shows a timeline of all valid horizons, highlighting the "best" (shortest confident) one.

    Returns:
        HTML fragment with multi-horizon prediction display
    """
    try:
        # Run blocking operation in thread pool
        result = await asyncio.to_thread(
            prediction_service.get_channel_projection, force_refresh=False
        )

        projections = result['projections']
        best = result['best_horizon']
        current_price = result['current_price']
        raw_confidence = result['raw_confidence']
        timestamp = result['timestamp'].strftime('%Y-%m-%d %H:%M:%S')

        # Extract breakout prediction if available
        breakout = result.get('breakout', None)
        breakout_html = ""
        if breakout:
            bo_prob = breakout['probability']
            bo_dir = breakout['direction_label']
            bo_bars = breakout['bars_until']
            bo_conf = breakout['confidence']
            bo_trained = breakout.get('is_trained', False)

            # Color-code breakout probability
            if bo_prob >= 0.7:
                bo_color = "red" if bo_dir == "down" else "green"
                bo_icon = "🚨" if bo_dir == "down" else "🚀"
                bo_label = "HIGH"
            elif bo_prob >= 0.5:
                bo_color = "yellow"
                bo_icon = "⚠️"
                bo_label = "MODERATE"
            else:
                bo_color = "gray"
                bo_icon = "📊"
                bo_label = "LOW"

            trained_badge = "" if bo_trained else " <span class='text-xs text-gray-500'>(untrained)</span>"

            breakout_html = f"""
            <div class="bg-{bo_color}-900 bg-opacity-30 border border-{bo_color}-600 rounded-lg p-3 mb-4">
                <div class="flex justify-between items-center">
                    <div>
                        <span class="text-sm text-{bo_color}-400">{bo_icon} Channel Breakout Risk{trained_badge}</span>
                        <div class="text-lg font-bold text-{bo_color}-400">{bo_label} - {bo_prob:.0%} probability</div>
                    </div>
                    <div class="text-right">
                        <div class="text-xs text-gray-400">Direction</div>
                        <div class="text-lg font-bold text-{bo_color}-400">{bo_dir.upper()}</div>
                    </div>
                </div>
                <div class="text-xs text-gray-400 mt-2">
                    Est. ~{bo_bars:.0f} bars until breakout | Confidence: {bo_conf:.0%}
                </div>
            </div>
            """

        # Build horizon cards
        horizon_cards = ""
        horizon_labels = {
            15: "15min",
            30: "30min",
            60: "1 hour",
            120: "2 hours",
            240: "4 hours",
            1440: "24 hours"
        }

        if projections:
            for i, proj in enumerate(projections):
                horizon_min = proj['horizon_minutes']
                label = horizon_labels.get(horizon_min, f"{horizon_min}min")
                conf = proj['confidence']
                high_price = proj['predicted_high_price']
                low_price = proj['predicted_low_price']
                high_pct = proj['predicted_high']
                low_pct = proj['predicted_low']

                # Highlight the best (first/shortest) horizon
                is_best = (i == 0)
                border_class = "border-blue-500 border-2" if is_best else "border-gray-600"
                bg_class = "bg-blue-900 bg-opacity-30" if is_best else "bg-gray-800"
                badge = '<span class="text-xs bg-blue-600 text-white px-2 py-1 rounded ml-2">BEST</span>' if is_best else ""

                # Confidence color
                if conf >= 0.75:
                    conf_color = "text-green-400"
                elif conf >= 0.65:
                    conf_color = "text-yellow-400"
                else:
                    conf_color = "text-orange-400"

                horizon_cards += f"""
                <div class="{bg_class} {border_class} rounded-lg p-4">
                    <div class="flex justify-between items-center mb-2">
                        <span class="text-lg font-bold text-blue-300">{label}{badge}</span>
                        <span class="text-sm {conf_color}">{conf:.0%} conf</span>
                    </div>
                    <div class="grid grid-cols-2 gap-2 text-sm">
                        <div class="bg-green-900 bg-opacity-40 rounded p-2">
                            <div class="text-green-400 text-xs">High</div>
                            <div class="text-green-400 font-bold">${high_price:.2f}</div>
                            <div class="text-green-400 text-xs">{high_pct:+.2f}%</div>
                        </div>
                        <div class="bg-red-900 bg-opacity-40 rounded p-2">
                            <div class="text-red-400 text-xs">Low</div>
                            <div class="text-red-400 font-bold">${low_price:.2f}</div>
                            <div class="text-red-400 text-xs">{low_pct:+.2f}%</div>
                        </div>
                    </div>
                </div>
                """

            # Best horizon summary
            best_label = horizon_labels.get(best['horizon_minutes'], f"{best['horizon_minutes']}min")
            best_conf = best['confidence']
            best_high = best['predicted_high_price']
            best_low = best['predicted_low_price']

            return f"""
            <div id="channel-projection" class="bg-gray-800 rounded-lg p-6 border border-gray-700">
                <div class="flex justify-between items-start mb-4">
                    <div>
                        <h2 class="text-2xl font-bold text-blue-400">Dynamic Horizon Prediction</h2>
                        <p class="text-sm text-gray-400">{timestamp}</p>
                    </div>
                    <div class="text-right">
                        <div class="text-sm text-gray-400">Current Price</div>
                        <div class="text-2xl font-bold">${current_price:.2f}</div>
                    </div>
                </div>

                {breakout_html}

                <!-- Best Horizon Highlight -->
                <div class="bg-blue-900 bg-opacity-30 border border-blue-600 rounded-lg p-4 mb-4">
                    <div class="flex justify-between items-center">
                        <div>
                            <span class="text-sm text-blue-300">Best Prediction Available</span>
                            <div class="text-3xl font-bold text-blue-400">{best_label}</div>
                        </div>
                        <div class="text-right">
                            <div class="text-sm text-gray-400">Confidence</div>
                            <div class="text-2xl font-bold text-green-400">{best_conf:.0%}</div>
                        </div>
                    </div>
                    <div class="grid grid-cols-2 gap-4 mt-3">
                        <div class="text-center">
                            <div class="text-sm text-green-400">Target High</div>
                            <div class="text-xl font-bold text-green-400">${best_high:.2f}</div>
                        </div>
                        <div class="text-center">
                            <div class="text-sm text-red-400">Target Low</div>
                            <div class="text-xl font-bold text-red-400">${best_low:.2f}</div>
                        </div>
                    </div>
                </div>

                <!-- All Valid Horizons -->
                <div class="mb-4">
                    <h3 class="text-sm text-gray-400 mb-2">All Valid Horizons ({len(projections)} available)</h3>
                    <div class="grid grid-cols-2 md:grid-cols-3 gap-3">
                        {horizon_cards}
                    </div>
                </div>

                <div class="text-xs text-gray-500">
                    <p>Raw model confidence: {raw_confidence:.0%} | Decay: e^(-horizon/60min)</p>
                </div>
            </div>
            """

        else:
            # No projections meet confidence threshold
            return f"""
            <div id="channel-projection" class="bg-gray-800 rounded-lg p-6 border border-yellow-700">
                <div class="flex justify-between items-start mb-4">
                    <div>
                        <h2 class="text-2xl font-bold text-yellow-400">Low Confidence</h2>
                        <p class="text-sm text-gray-400">{timestamp}</p>
                    </div>
                    <div class="text-right">
                        <div class="text-sm text-gray-400">Current Price</div>
                        <div class="text-2xl font-bold">${current_price:.2f}</div>
                    </div>
                </div>

                {breakout_html}

                <div class="bg-yellow-900 bg-opacity-30 border border-yellow-700 rounded-lg p-4">
                    <p class="text-yellow-400">
                        ⚠️ Model confidence ({raw_confidence:.0%}) is too low for reliable predictions.
                    </p>
                    <p class="text-sm text-gray-400 mt-2">
                        After applying time decay, no horizon meets the 60% confidence threshold.
                        Consider waiting for more favorable market conditions.
                    </p>
                </div>

                <div class="mt-4 text-xs text-gray-500">
                    <p>Raw model confidence: {raw_confidence:.0%} | Required after decay: ≥60%</p>
                </div>
            </div>
            """

    except Exception as e:
        logger.exception("Error generating channel projection")
        return f"""
        <div id="channel-projection" class="bg-gray-800 rounded-lg p-6 border border-red-700">
            <p class="text-red-400">Error generating channel projection. Please try again.</p>
        </div>
        """


@router.post("/channel/generate", response_class=HTMLResponse)
@limiter.limit("1/minute")
async def generate_channel_projection(request: Request):
    """
    Force generation of new channel projection (ignores cache).
    Rate limited: 1 request per minute

    Returns:
        HTML with fresh channel projection
    """
    try:
        logger.info("Generating fresh channel projection (user requested)...")
        result = await asyncio.to_thread(
            prediction_service.get_channel_projection, force_refresh=True
        )

        # Reuse the same rendering logic as GET /channel
        # For simplicity, redirect to the GET endpoint's response format
        projections = result['projections']
        best = result['best_horizon']
        current_price = result['current_price']
        raw_confidence = result['raw_confidence']
        timestamp = result['timestamp'].strftime('%Y-%m-%d %H:%M:%S')

        horizon_labels = {
            15: "15min", 30: "30min", 60: "1 hour",
            120: "2 hours", 240: "4 hours", 1440: "24 hours"
        }

        if projections:
            best_label = horizon_labels.get(best['horizon_minutes'], f"{best['horizon_minutes']}min")
            return f"""
            <div id="channel-projection" class="bg-gray-800 rounded-lg p-6 border border-green-700">
                <div class="bg-green-900 bg-opacity-20 border border-green-700 rounded-lg p-2 mb-4">
                    <p class="text-sm text-green-400">✅ Fresh channel projection generated!</p>
                </div>
                <div class="flex justify-between items-start mb-4">
                    <div>
                        <h2 class="text-2xl font-bold text-blue-400">Dynamic Horizon Prediction</h2>
                        <p class="text-sm text-gray-400">{timestamp}</p>
                    </div>
                    <div class="text-right">
                        <div class="text-sm text-gray-400">Current Price</div>
                        <div class="text-2xl font-bold">${current_price:.2f}</div>
                    </div>
                </div>

                <div class="bg-blue-900 bg-opacity-30 border border-blue-600 rounded-lg p-4">
                    <div class="flex justify-between items-center">
                        <div>
                            <span class="text-sm text-blue-300">Best Prediction</span>
                            <div class="text-3xl font-bold text-blue-400">{best_label}</div>
                        </div>
                        <div class="text-right">
                            <div class="text-sm text-gray-400">Confidence</div>
                            <div class="text-2xl font-bold text-green-400">{best['confidence']:.0%}</div>
                        </div>
                    </div>
                    <div class="grid grid-cols-2 gap-4 mt-3">
                        <div class="text-center">
                            <div class="text-sm text-green-400">Target High</div>
                            <div class="text-xl font-bold text-green-400">${best['predicted_high_price']:.2f}</div>
                        </div>
                        <div class="text-center">
                            <div class="text-sm text-red-400">Target Low</div>
                            <div class="text-xl font-bold text-red-400">${best['predicted_low_price']:.2f}</div>
                        </div>
                    </div>
                </div>

                <p class="text-xs text-gray-500 mt-4">
                    {len(projections)} valid horizons | Raw confidence: {raw_confidence:.0%}
                </p>
            </div>
            """
        else:
            return f"""
            <div id="channel-projection" class="bg-yellow-900 bg-opacity-30 border border-yellow-700 rounded-lg p-6">
                <p class="text-yellow-400 font-bold">⚠️ Low Confidence</p>
                <p class="text-sm text-gray-400 mt-2">
                    Model confidence ({raw_confidence:.0%}) too low after time decay.
                </p>
            </div>
            """

    except Exception as e:
        logger.exception("Failed to generate channel projection")
        return """
        <div id="channel-projection" class="bg-red-900 bg-opacity-30 border border-red-700 rounded-lg p-6">
            <p class="text-red-400 font-bold">❌ Failed to generate channel projection</p>
            <p class="text-sm text-gray-400 mt-2">Please try again later.</p>
        </div>
        """


@router.get("/setups", response_class=HTMLResponse)
async def get_trade_setups(request: Request):
    """
    Get multi-timeframe trade setups based on channel analysis.

    Analyzes channels across Scalp, Intraday, Swing, and Position timeframes
    and returns setups that meet confidence thresholds.

    Returns:
        HTML fragment with trade setup cards
    """
    try:
        # Run blocking operation in thread pool
        result = await asyncio.to_thread(
            prediction_service.get_trade_setups, force_refresh=False
        )

        setups = result['setups']
        best_setup = result['best_setup']
        current_price = result['current_price']
        timestamp = result['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
        model_prediction = result['model_prediction']

        # Build setup cards
        setup_cards = ""

        # Type-specific styling
        type_styles = {
            'scalp': {'color': 'purple', 'icon': '⚡'},
            'intraday': {'color': 'blue', 'icon': '📊'},
            'swing': {'color': 'green', 'icon': '📈'},
            'position': {'color': 'orange', 'icon': '🎯'}
        }

        for setup in setups:
            style = type_styles.get(setup['type'], {'color': 'gray', 'icon': '📌'})
            color = style['color']
            icon = style['icon']

            # Confidence color
            conf = setup['confidence']
            if conf >= 75:
                conf_color = "green"
                conf_badge = "HIGH"
            elif conf >= 55:
                conf_color = "yellow"
                conf_badge = "MEDIUM"
            else:
                conf_color = "orange"
                conf_badge = "LOW"

            # Direction indicator
            dir_map = {
                'bullish': ('↗️', 'green', 'BULLISH'),
                'bearish': ('↘️', 'red', 'BEARISH'),
                'neutral': ('→', 'gray', 'NEUTRAL')
            }
            dir_icon, dir_color, dir_label = dir_map.get(setup['direction'], ('→', 'gray', 'NEUTRAL'))

            # Is this the best setup?
            is_best = (best_setup and setup['type'] == best_setup['type'])
            best_badge = '<span class="text-xs bg-blue-600 text-white px-2 py-1 rounded ml-2">BEST</span>' if is_best else ''
            border_class = "border-blue-500 border-2" if is_best else f"border-{color}-600"

            setup_cards += f"""
            <div class="bg-gray-800 {border_class} rounded-lg p-4 mb-3">
                <div class="flex justify-between items-start mb-3">
                    <div>
                        <span class="text-lg font-bold text-{color}-400">{icon} {setup['label']}{best_badge}</span>
                        <div class="text-xs text-gray-400">{setup['description']}</div>
                        <div class="text-xs text-gray-500">{setup['channel_timeframe']} channel (R²={setup['r_squared']:.2f})</div>
                    </div>
                    <div class="text-right">
                        <div class="text-xs text-gray-400">Confidence</div>
                        <div class="text-xl font-bold text-{conf_color}-400">{conf:.0f}%</div>
                        <div class="text-xs text-{conf_color}-400">{conf_badge}</div>
                    </div>
                </div>

                <div class="grid grid-cols-2 gap-3 mb-3">
                    <div class="bg-green-900 bg-opacity-30 rounded-lg p-3 border border-green-700">
                        <div class="text-xs text-green-400">Target High</div>
                        <div class="text-lg font-bold text-green-400">${setup['high']:.2f}</div>
                        <div class="text-xs text-green-400">+{((setup['high']/current_price - 1) * 100):.2f}%</div>
                    </div>
                    <div class="bg-red-900 bg-opacity-30 rounded-lg p-3 border border-red-700">
                        <div class="text-xs text-red-400">Target Low</div>
                        <div class="text-lg font-bold text-red-400">${setup['low']:.2f}</div>
                        <div class="text-xs text-red-400">{((setup['low']/current_price - 1) * 100):.2f}%</div>
                    </div>
                </div>

                <div class="flex justify-between items-center text-sm">
                    <div>
                        <span class="text-gray-400">Duration:</span>
                        <span class="text-white">{setup['duration']}</span>
                    </div>
                    <div>
                        <span class="text-{dir_color}-400">{dir_icon} {dir_label}</span>
                        <span class="text-gray-500 text-xs ml-1">({setup['slope_pct']:+.2f}%/bar)</span>
                    </div>
                </div>

                <div class="mt-2 text-xs text-gray-500">
                    {setup['risk_note']} | Position: {setup['position_in_channel']:.0%} in channel
                </div>
            </div>
            """

        # No setups case
        if not setups:
            setup_cards = """
            <div class="bg-yellow-900 bg-opacity-30 border border-yellow-700 rounded-lg p-4">
                <p class="text-yellow-400">⚠️ No valid trade setups found</p>
                <p class="text-sm text-gray-400 mt-2">
                    All channels are below the R² threshold (0.40).
                    Wait for more defined price action.
                </p>
            </div>
            """

        # Model prediction comparison
        model_conf = model_prediction['confidence']
        model_high = model_prediction['predicted_high']
        model_low = model_prediction['predicted_low']
        model_high_price = current_price * (1 + model_high / 100)
        model_low_price = current_price * (1 + model_low / 100)

        return f"""
        <div id="trade-setups" class="bg-gray-900 rounded-lg p-6">
            <div class="flex justify-between items-start mb-4">
                <div>
                    <h2 class="text-2xl font-bold text-blue-400">Multi-Timeframe Trade Setups</h2>
                    <p class="text-sm text-gray-400">{timestamp}</p>
                    <p class="text-xs text-gray-500">{len(setups)} setup(s) found</p>
                </div>
                <div class="text-right">
                    <div class="text-sm text-gray-400">TSLA Price</div>
                    <div class="text-2xl font-bold">${current_price:.2f}</div>
                </div>
            </div>

            <!-- Setup Cards -->
            <div class="mb-4">
                {setup_cards}
            </div>

            <!-- Model Prediction Comparison -->
            <div class="bg-gray-800 border border-gray-600 rounded-lg p-4 mt-4">
                <div class="text-sm text-gray-400 mb-2">Model Fused Prediction (24h horizon)</div>
                <div class="grid grid-cols-3 gap-3 text-center">
                    <div>
                        <div class="text-xs text-gray-500">High</div>
                        <div class="text-green-400 font-bold">${model_high_price:.2f}</div>
                        <div class="text-xs text-green-400">{model_high:+.2f}%</div>
                    </div>
                    <div>
                        <div class="text-xs text-gray-500">Low</div>
                        <div class="text-red-400 font-bold">${model_low_price:.2f}</div>
                        <div class="text-xs text-red-400">{model_low:+.2f}%</div>
                    </div>
                    <div>
                        <div class="text-xs text-gray-500">Confidence</div>
                        <div class="text-blue-400 font-bold">{model_conf:.0%}</div>
                    </div>
                </div>
            </div>
        </div>
        """

    except Exception as e:
        logger.exception("Error generating trade setups")
        return f"""
        <div id="trade-setups" class="bg-gray-800 rounded-lg p-6 border border-red-700">
            <p class="text-red-400">Error generating trade setups. Please try again.</p>
            <p class="text-xs text-gray-500 mt-2">{escape(str(e))}</p>
        </div>
        """


@router.post("/setups/generate", response_class=HTMLResponse)
@limiter.limit("1/minute")
async def generate_trade_setups(request: Request):
    """
    Force generation of new trade setups (ignores cache).
    Rate limited: 1 request per minute

    Returns:
        HTML with fresh trade setups
    """
    try:
        logger.info("Generating fresh trade setups (user requested)...")
        result = await asyncio.to_thread(
            prediction_service.get_trade_setups, force_refresh=True
        )

        setups = result['setups']
        current_price = result['current_price']
        timestamp = result['timestamp'].strftime('%Y-%m-%d %H:%M:%S')

        if setups:
            best = setups[0]
            return f"""
            <div id="trade-setups" class="bg-gray-800 rounded-lg p-6 border border-green-700">
                <div class="bg-green-900 bg-opacity-20 border border-green-700 rounded-lg p-2 mb-4">
                    <p class="text-sm text-green-400">✅ Fresh trade setups generated!</p>
                </div>
                <div class="flex justify-between items-start mb-4">
                    <div>
                        <h2 class="text-2xl font-bold text-blue-400">Trade Setups</h2>
                        <p class="text-sm text-gray-400">{timestamp}</p>
                    </div>
                    <div class="text-right">
                        <div class="text-sm text-gray-400">TSLA</div>
                        <div class="text-2xl font-bold">${current_price:.2f}</div>
                    </div>
                </div>
                <div class="bg-blue-900 bg-opacity-30 border border-blue-600 rounded-lg p-4">
                    <div class="flex justify-between items-center">
                        <div>
                            <span class="text-sm text-blue-300">Best Setup</span>
                            <div class="text-2xl font-bold text-blue-400">{best['label']}</div>
                            <div class="text-xs text-gray-400">{best['channel_timeframe']} channel</div>
                        </div>
                        <div class="text-right">
                            <div class="text-sm text-gray-400">Confidence</div>
                            <div class="text-2xl font-bold text-green-400">{best['confidence']:.0f}%</div>
                        </div>
                    </div>
                    <div class="grid grid-cols-2 gap-4 mt-3">
                        <div class="text-center">
                            <div class="text-sm text-green-400">High</div>
                            <div class="text-xl font-bold text-green-400">${best['high']:.2f}</div>
                        </div>
                        <div class="text-center">
                            <div class="text-sm text-red-400">Low</div>
                            <div class="text-xl font-bold text-red-400">${best['low']:.2f}</div>
                        </div>
                    </div>
                </div>
                <p class="text-xs text-gray-500 mt-4">{len(setups)} total setup(s) | Duration: {best['duration']}</p>
            </div>
            """
        else:
            return f"""
            <div id="trade-setups" class="bg-yellow-900 bg-opacity-30 border border-yellow-700 rounded-lg p-6">
                <p class="text-yellow-400 font-bold">⚠️ No Valid Trade Setups</p>
                <p class="text-sm text-gray-400 mt-2">
                    All channels below R² threshold. Current price: ${current_price:.2f}
                </p>
            </div>
            """

    except Exception as e:
        logger.exception("Failed to generate trade setups")
        return """
        <div id="trade-setups" class="bg-red-900 bg-opacity-30 border border-red-700 rounded-lg p-6">
            <p class="text-red-400 font-bold">❌ Failed to generate trade setups</p>
            <p class="text-sm text-gray-400 mt-2">Please try again later.</p>
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
