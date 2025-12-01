"""
Charts API Router - Plotly.js data endpoints
"""
from fastapi import APIRouter
from typing import Dict
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from backend.app.services.performance_service import performance_service

router = APIRouter()


@router.get("/pnl-over-time")
async def get_pnl_chart_data() -> Dict:
    """
    Get cumulative P&L over time for Plotly chart

    Returns:
        JSON with Plotly trace data
    """
    data = performance_service.get_pnl_over_time()

    if not data['dates']:
        return {
            'data': [],
            'layout': {
                'title': 'No trades yet',
                'paper_bgcolor': 'rgba(0,0,0,0)',
                'plot_bgcolor': 'rgba(0,0,0,0)',
                'font': {'color': '#9CA3AF'}
            }
        }

    # Create Plotly trace
    trace = {
        'x': data['dates'],
        'y': data['cumulative_pnl'],
        'type': 'scatter',
        'mode': 'lines+markers',
        'name': 'Cumulative P&L',
        'line': {
            'color': '#10B981' if data['cumulative_pnl'][-1] > 0 else '#EF4444',
            'width': 3
        },
        'marker': {
            'size': 6,
            'color': '#60A5FA'
        },
        'fill': 'tozeroy',
        'fillcolor': 'rgba(16, 185, 129, 0.1)' if data['cumulative_pnl'][-1] > 0 else 'rgba(239, 68, 68, 0.1)'
    }

    layout = {
        'title': {
            'text': 'Cumulative P&L Over Time',
            'font': {'size': 18, 'color': '#60A5FA'}
        },
        'xaxis': {
            'title': 'Date',
            'color': '#9CA3AF',
            'gridcolor': '#374151'
        },
        'yaxis': {
            'title': 'P&L ($)',
            'color': '#9CA3AF',
            'gridcolor': '#374151',
            'zeroline': True,
            'zerolinecolor': '#6B7280',
            'zerolinewidth': 2
        },
        'paper_bgcolor': '#1F2937',
        'plot_bgcolor': '#111827',
        'font': {'color': '#9CA3AF'},
        'hovermode': 'x unified',
        'showlegend': True,
        'height': 400
    }

    return {
        'data': [trace],
        'layout': layout
    }


@router.get("/returns-distribution")
async def get_returns_histogram() -> Dict:
    """
    Get distribution of returns for Plotly histogram

    Returns:
        JSON with Plotly histogram data
    """
    data = performance_service.get_returns_distribution()

    if not data['returns']:
        return {
            'data': [],
            'layout': {
                'title': 'No trades yet',
                'paper_bgcolor': 'rgba(0,0,0,0)',
                'plot_bgcolor': 'rgba(0,0,0,0)',
                'font': {'color': '#9CA3AF'}
            }
        }

    trace = {
        'x': data['returns'],
        'y': data['counts'],
        'type': 'bar',
        'name': 'Returns',
        'marker': {
            'color': '#60A5FA',
            'line': {'color': '#3B82F6', 'width': 1}
        }
    }

    layout = {
        'title': {
            'text': 'Distribution of Returns',
            'font': {'size': 18, 'color': '#60A5FA'}
        },
        'xaxis': {
            'title': 'Return (%)',
            'color': '#9CA3AF',
            'gridcolor': '#374151'
        },
        'yaxis': {
            'title': 'Frequency',
            'color': '#9CA3AF',
            'gridcolor': '#374151'
        },
        'paper_bgcolor': '#1F2937',
        'plot_bgcolor': '#111827',
        'font': {'color': '#9CA3AF'},
        'showlegend': False,
        'height': 400
    }

    return {
        'data': [trace],
        'layout': layout
    }
