"""
Prediction database for logging and tracking ML model performance
Uses SQLAlchemy for flexible database backend (SQLite or PostgreSQL)
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import sys

# Add parent directory to path
parent_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(parent_dir))

import config
from .base import PredictionDatabase

Base = declarative_base()


class Prediction(Base):
    """SQLAlchemy model for predictions table"""
    __tablename__ = 'predictions'

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Timing
    timestamp = Column(DateTime, nullable=False, index=True)
    prediction_timestamp = Column(DateTime, nullable=False)
    target_timestamp = Column(DateTime, nullable=False)  # When prediction is for
    simulation_date = Column(DateTime, nullable=True, index=True)  # Historical date being backtested (for alignment)

    # Symbol and timeframe
    symbol = Column(String(10), nullable=False, index=True)
    timeframe = Column(String(10), nullable=False, index=True)

    # Multi-scale ensemble fields
    model_timeframe = Column(String(20), nullable=True, index=True)  # e.g., '15min', '1hour', 'ensemble'
    is_ensemble = Column(Boolean, default=False, index=True)
    news_enabled = Column(Boolean, default=False)

    # Predictions (now in percentage terms)
    predicted_high = Column(Float, nullable=False)  # % change from current price
    predicted_low = Column(Float, nullable=False)   # % change from current price
    predicted_center = Column(Float, nullable=False)
    predicted_range = Column(Float, nullable=False)
    confidence = Column(Float, nullable=False)

    # Current price (needed for percentage → absolute conversion)
    current_price = Column(Float, nullable=True)

    # Actuals (filled in later)
    actual_high = Column(Float, nullable=True)
    actual_low = Column(Float, nullable=True)
    actual_center = Column(Float, nullable=True)
    has_actuals = Column(Boolean, default=False, index=True)

    # Errors (calculated after actuals filled)
    error_high = Column(Float, nullable=True)
    error_low = Column(Float, nullable=True)
    error_center = Column(Float, nullable=True)
    absolute_error = Column(Float, nullable=True)  # Average of high/low errors

    # Context features
    channel_position = Column(Float, nullable=True)
    rsi_value = Column(Float, nullable=True)
    spy_correlation = Column(Float, nullable=True)

    # Events
    has_earnings = Column(Boolean, default=False)
    has_macro_event = Column(Boolean, default=False)
    event_type = Column(String(50), nullable=True)

    # Sub-model predictions (for ensemble analysis)
    sub_pred_15min_high = Column(Float, nullable=True)
    sub_pred_15min_low = Column(Float, nullable=True)
    sub_pred_15min_conf = Column(Float, nullable=True)
    sub_pred_1hour_high = Column(Float, nullable=True)
    sub_pred_1hour_low = Column(Float, nullable=True)
    sub_pred_1hour_conf = Column(Float, nullable=True)
    sub_pred_4hour_high = Column(Float, nullable=True)
    sub_pred_4hour_low = Column(Float, nullable=True)
    sub_pred_4hour_conf = Column(Float, nullable=True)
    sub_pred_daily_high = Column(Float, nullable=True)
    sub_pred_daily_low = Column(Float, nullable=True)
    sub_pred_daily_conf = Column(Float, nullable=True)

    # Model metadata
    model_version = Column(String(50), nullable=True)
    feature_dim = Column(Integer, nullable=True)

    def __repr__(self):
        return f"<Prediction(id={self.id}, {self.symbol} @ {self.timestamp}, " \
               f"pred=[{self.predicted_low:.2f}-{self.predicted_high:.2f}], " \
               f"confidence={self.confidence:.2f})>"


class SQLitePredictionDB(PredictionDatabase):
    """
    SQLite implementation of prediction database
    Suitable for local development and single-machine deployment
    """

    def __init__(self, db_path: str = None):
        self.db_path = db_path or config.ML_DB_PATH

        # Create database directory if needed
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        # Create engine and session
        self.engine = create_engine(f'sqlite:///{self.db_path}')
        Base.metadata.create_all(self.engine)

        Session = sessionmaker(bind=self.engine)
        self.session = Session()

        print(f"Prediction database initialized at {self.db_path}")

    def log_prediction(self, prediction: Dict[str, Any]) -> int:
        """
        Log a new prediction to database
        Returns prediction ID
        """
        pred_record = Prediction(
            timestamp=datetime.now(),
            prediction_timestamp=prediction.get('prediction_timestamp', datetime.now()),
            target_timestamp=prediction['target_timestamp'],
            simulation_date=prediction.get('simulation_date'),  # Historical date being backtested
            symbol=prediction.get('symbol', 'TSLA'),
            timeframe=prediction.get('timeframe', '1h'),
            predicted_high=float(prediction['predicted_high']),
            predicted_low=float(prediction['predicted_low']),
            predicted_center=float(prediction['predicted_center']),
            predicted_range=float(prediction['predicted_range']),
            confidence=float(prediction['confidence']),
            current_price=prediction.get('current_price'),
            channel_position=prediction.get('channel_position'),
            rsi_value=prediction.get('rsi_value'),
            spy_correlation=prediction.get('spy_correlation'),
            has_earnings=prediction.get('has_earnings', False),
            has_macro_event=prediction.get('has_macro_event', False),
            event_type=prediction.get('event_type'),
            model_version=prediction.get('model_version'),
            feature_dim=prediction.get('feature_dim'),
            model_timeframe=prediction.get('model_timeframe'),
            is_ensemble=prediction.get('is_ensemble', False),
            news_enabled=prediction.get('news_enabled', False)
        )

        self.session.add(pred_record)
        self.session.commit()

        return pred_record.id

    def get_prediction(self, prediction_id: int) -> Dict[str, Any]:
        """Retrieve a specific prediction"""
        pred = self.session.query(Prediction).filter(Prediction.id == prediction_id).first()

        if not pred:
            raise ValueError(f"Prediction {prediction_id} not found")

        return {
            'id': pred.id,
            'timestamp': pred.timestamp,
            'symbol': pred.symbol,
            'timeframe': pred.timeframe,
            'predicted_high': pred.predicted_high,
            'predicted_low': pred.predicted_low,
            'confidence': pred.confidence,
            'actual_high': pred.actual_high,
            'actual_low': pred.actual_low,
            'error_high': pred.error_high,
            'error_low': pred.error_low,
            'has_actuals': pred.has_actuals
        }

    def update_actual(self, prediction_id: int, actual_high: float, actual_low: float):
        """Update prediction with actual values and calculate errors"""
        pred = self.session.query(Prediction).filter(Prediction.id == prediction_id).first()

        if not pred:
            raise ValueError(f"Prediction {prediction_id} not found")

        # Validate current_price exists (needed for error calculation)
        if pred.current_price is None:
            raise ValueError(f"Prediction {prediction_id} has current_price=None! Bug in log_prediction() - not extracting current_price from prediction dict")

        # Update actuals
        pred.actual_high = actual_high
        pred.actual_low = actual_low
        pred.actual_center = (actual_high + actual_low) / 2
        pred.has_actuals = True

        # Calculate errors (compare percentage predictions to percentage actuals)
        # Convert actual prices to percentage changes from current price
        actual_high_pct = (actual_high - pred.current_price) / pred.current_price * 100
        actual_low_pct = (actual_low - pred.current_price) / pred.current_price * 100
        actual_center_pct = (pred.actual_center - pred.current_price) / pred.current_price * 100

        # Errors are now in percentage points (e.g., predicted +2.5% but actual was +3.2% = 0.7pp error)
        pred.error_high = abs(pred.predicted_high - actual_high_pct)
        pred.error_low = abs(pred.predicted_low - actual_low_pct)
        pred.error_center = abs(pred.predicted_center - actual_center_pct)
        pred.absolute_error = (pred.error_high + pred.error_low) / 2

        self.session.commit()

        print(f"Updated prediction {prediction_id} with actuals: "
              f"high={actual_high:.2f} (err={pred.error_high:.2f}%), "
              f"low={actual_low:.2f} (err={pred.error_low:.2f}%)")

    def get_accuracy_metrics(self, timeframe: str = None, limit: int = 1000) -> Dict[str, float]:
        """
        Calculate accuracy metrics for predictions with actuals
        Returns mean errors, confidence calibration, etc.
        """
        query = self.session.query(Prediction).filter(Prediction.has_actuals == True)

        if timeframe:
            query = query.filter(Prediction.timeframe == timeframe)

        predictions = query.order_by(Prediction.timestamp.desc()).limit(limit).all()

        if not predictions:
            return {
                'num_predictions': 0,
                'mean_absolute_error': 0.0,
                'mean_error_high': 0.0,
                'mean_error_low': 0.0,
                'mean_confidence': 0.0,
                'accuracy_by_confidence': {}
            }

        # Calculate metrics
        errors_high = [p.error_high for p in predictions]
        errors_low = [p.error_low for p in predictions]
        errors_abs = [p.absolute_error for p in predictions]
        confidences = [p.confidence for p in predictions]

        # Check for None values (indicates bugs in current_price or update_actual)
        none_count_high = sum(1 for e in errors_high if e is None)
        none_count_low = sum(1 for e in errors_low if e is None)
        none_count_abs = sum(1 for e in errors_abs if e is None)

        if none_count_abs > 0:
            raise ValueError(
                f"Found {none_count_abs}/{len(predictions)} predictions with None errors! "
                f"({none_count_high} high, {none_count_low} low). "
                f"This indicates current_price was not stored. Check log_prediction() extraction."
            )

        metrics = {
            'num_predictions': len(predictions),
            'mean_absolute_error': np.mean(errors_abs),
            'median_absolute_error': np.median(errors_abs),
            'std_absolute_error': np.std(errors_abs),
            'mean_error_high': np.mean(errors_high),
            'mean_error_low': np.mean(errors_low),
            'mean_confidence': np.mean(confidences),
            'median_confidence': np.median(confidences)
        }

        # Accuracy by confidence bins
        confidence_bins = [(0, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 1.0)]
        for low, high in confidence_bins:
            bin_preds = [p for p in predictions if low <= p.confidence < high]
            if bin_preds:
                bin_error = np.mean([p.absolute_error for p in bin_preds])
                metrics[f'error_confidence_{low:.1f}_{high:.1f}'] = bin_error

        return metrics

    def get_error_patterns(self, limit: int = 100) -> pd.DataFrame:
        """
        Get predictions with largest errors for analysis
        Returns DataFrame for easy analysis
        """
        predictions = self.session.query(Prediction)\
            .filter(Prediction.has_actuals == True)\
            .order_by(Prediction.absolute_error.desc())\
            .limit(limit)\
            .all()

        if not predictions:
            return pd.DataFrame()

        data = []
        for p in predictions:
            data.append({
                'id': p.id,
                'timestamp': p.timestamp,
                'symbol': p.symbol,
                'timeframe': p.timeframe,
                'predicted_high': p.predicted_high,
                'predicted_low': p.predicted_low,
                'actual_high': p.actual_high,
                'actual_low': p.actual_low,
                'error_high': p.error_high,
                'error_low': p.error_low,
                'absolute_error': p.absolute_error,
                'confidence': p.confidence,
                'channel_position': p.channel_position,
                'rsi_value': p.rsi_value,
                'has_earnings': p.has_earnings,
                'has_macro_event': p.has_macro_event,
                'event_type': p.event_type
            })

        return pd.DataFrame(data)

    def get_predictions_needing_update(self, hours_elapsed: int = 24) -> List[Dict]:
        """
        Get predictions that are past their target time and need actual values
        Used for batch updating actuals
        """
        from datetime import timedelta
        cutoff_time = datetime.now() - timedelta(hours=hours_elapsed)

        predictions = self.session.query(Prediction)\
            .filter(Prediction.has_actuals == False)\
            .filter(Prediction.target_timestamp <= cutoff_time)\
            .all()

        return [
            {
                'id': p.id,
                'target_timestamp': p.target_timestamp,
                'symbol': p.symbol,
                'timeframe': p.timeframe
            }
            for p in predictions
        ]

    def get_recent_predictions(self, hours: int = 24, limit: int = 100) -> pd.DataFrame:
        """Get recent predictions for monitoring"""
        from datetime import timedelta
        cutoff_time = datetime.now() - timedelta(hours=hours)

        predictions = self.session.query(Prediction)\
            .filter(Prediction.timestamp >= cutoff_time)\
            .order_by(Prediction.timestamp.desc())\
            .limit(limit)\
            .all()

        if not predictions:
            return pd.DataFrame()

        data = []
        for p in predictions:
            data.append({
                'id': p.id,
                'timestamp': p.timestamp,
                'symbol': p.symbol,
                'predicted_high': p.predicted_high,
                'predicted_low': p.predicted_low,
                'confidence': p.confidence,
                'has_actuals': p.has_actuals,
                'error': p.absolute_error if p.has_actuals else None
            })

        return pd.DataFrame(data)

    def close(self):
        """Close database session"""
        self.session.close()
