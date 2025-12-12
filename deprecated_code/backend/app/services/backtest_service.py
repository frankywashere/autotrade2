"""
Backtesting Simulation Service

Quick win: Uses existing predictions.db with has_actuals=True
Compares predicted_high/low vs actual_high/low to calculate performance
"""
import pandas as pd
import numpy as np
from datetime import date, datetime
from typing import Dict, List, Optional
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from backend.app.models.database import Prediction, Backtest, db


class BacktestService:
    """
    Backtesting simulation using historical predictions

    Phase 1: Simple simulation (no order execution modeling)
    - Checks if predicted_high/low were achieved
    - Calculates win rate, total P&L, Sharpe ratio
    - Uses existing predictions.db data
    """

    def run_simulation(
        self,
        name: str,
        start_date: date,
        end_date: date,
        confidence_threshold: float = 0.7,
        layer: Optional[str] = None
    ) -> Dict:
        """
        Run backtest simulation on historical predictions

        Args:
            name: Backtest run name
            start_date: Start date for backtest
            end_date: End date for backtest
            confidence_threshold: Minimum confidence to trade (0-1)
            layer: Which layer to use (fast/medium/slow/fusion), None = fusion

        Returns:
            Backtest results dict
        """
        print(f"\nRunning backtest: {name}")
        print(f"  Period: {start_date} to {end_date}")
        print(f"  Confidence threshold: {confidence_threshold:.1%}")
        print(f"  Layer: {layer or 'fusion'}")

        # Query predictions with actuals
        query = (Prediction
                .select()
                .where(
                    (Prediction.has_actuals == True) &
                    (Prediction.timestamp >= start_date) &
                    (Prediction.timestamp <= end_date)
                )
                .order_by(Prediction.timestamp))

        predictions = list(query)
        print(f"  Found {len(predictions)} predictions with actuals")

        if len(predictions) == 0:
            raise ValueError("No predictions with actuals in this date range")

        # Select which predictions to use based on layer and confidence
        trades = []

        for pred in predictions:
            # Get layer-specific predictions or fusion
            # Map layer names to existing sub_pred columns
            if layer == '15min':
                pred_high = pred.sub_pred_15min_high
                pred_low = pred.sub_pred_15min_low
                conf = pred.sub_pred_15min_conf
            elif layer == '1hour':
                pred_high = pred.sub_pred_1hour_high
                pred_low = pred.sub_pred_1hour_low
                conf = pred.sub_pred_1hour_conf
            elif layer == '4hour':
                pred_high = pred.sub_pred_4hour_high
                pred_low = pred.sub_pred_4hour_low
                conf = pred.sub_pred_4hour_conf
            elif layer == 'daily':
                pred_high = pred.sub_pred_daily_high
                pred_low = pred.sub_pred_daily_low
                conf = pred.sub_pred_daily_conf
            else:  # fusion (default)
                pred_high = pred.predicted_high
                pred_low = pred.predicted_low
                conf = pred.confidence

            # Skip if confidence too low or missing data
            if conf is None or conf < confidence_threshold:
                continue

            if pred_high is None or pred_low is None:
                continue

            # Check if prediction was correct
            actual_high = pred.actual_high
            actual_low = pred.actual_low

            if actual_high is None or actual_low is None:
                continue

            # Calculate P&L (simplified - assumes we trade the predicted move)
            # If predicted_high was hit: profit = predicted_high
            # If predicted_low was hit first: loss = predicted_low
            # For now, assume we capture the average of what was achieved

            hit_high = actual_high >= (pred.current_price * (1 + pred_high / 100))
            hit_low = actual_low <= (pred.current_price * (1 + pred_low / 100))

            # Simple P&L calculation
            if hit_high:
                # Predicted move UP was achieved
                pnl_pct = min(pred_high, (actual_high / pred.current_price - 1) * 100)
            elif hit_low:
                # Predicted move DOWN was achieved
                pnl_pct = max(pred_low, (actual_low / pred.current_price - 1) * 100)
            else:
                # Neither target hit - neutral
                pnl_pct = 0

            trades.append({
                'timestamp': pred.timestamp,
                'entry_price': pred.current_price,
                'predicted_high': pred_high,
                'predicted_low': pred_low,
                'actual_high': actual_high,
                'actual_low': actual_low,
                'hit_high': hit_high,
                'hit_low': hit_low,
                'pnl_pct': pnl_pct,
                'confidence': conf
            })

        if len(trades) == 0:
            raise ValueError(f"No trades above confidence threshold {confidence_threshold:.1%}")

        # Calculate performance metrics
        df = pd.DataFrame(trades)

        total_trades = len(trades)
        winning_trades = len(df[df['pnl_pct'] > 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        total_pnl = df['pnl_pct'].sum()
        avg_pnl = df['pnl_pct'].mean()

        # Sharpe ratio (annualized, assuming 252 trading days)
        returns = df['pnl_pct'].values
        if len(returns) > 1:
            sharpe_ratio = (np.mean(returns) / np.std(returns)) * np.sqrt(252)
        else:
            sharpe_ratio = 0.0

        # Max drawdown (cumulative)
        cumulative_returns = (1 + df['pnl_pct'] / 100).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns / running_max) - 1
        max_drawdown = drawdowns.min() * 100  # Convert to percentage

        results = {
            'name': name,
            'start_date': start_date,
            'end_date': end_date,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': total_trades - winning_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_pnl': avg_pnl,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'config': {
                'confidence_threshold': confidence_threshold,
                'layer': layer or 'fusion'
            },
            'trades': trades[:100]  # Return first 100 trades for preview
        }

        # Save to database
        backtest_record = Backtest.create(
            name=name,
            start_date=start_date,
            end_date=end_date,
            total_trades=total_trades,
            win_rate=win_rate,
            total_pnl=total_pnl,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            config_json=str(results['config'])
        )

        results['id'] = backtest_record.id

        print(f"\n✓ Backtest complete:")
        print(f"  Total trades: {total_trades}")
        print(f"  Win rate: {win_rate:.1%}")
        print(f"  Total P&L: {total_pnl:+.2f}%")
        print(f"  Sharpe ratio: {sharpe_ratio:.2f}")
        print(f"  Max drawdown: {max_drawdown:.2f}%")

        return results

    def get_backtest(self, backtest_id: int) -> Optional[Dict]:
        """
        Get backtest results by ID

        Args:
            backtest_id: Backtest run ID

        Returns:
            Backtest results dict or None
        """
        try:
            backtest = Backtest.get_by_id(backtest_id)

            return {
                'id': backtest.id,
                'name': backtest.name,
                'run_date': backtest.run_date,
                'start_date': backtest.start_date,
                'end_date': backtest.end_date,
                'total_trades': backtest.total_trades,
                'win_rate': backtest.win_rate,
                'total_pnl': backtest.total_pnl,
                'sharpe_ratio': backtest.sharpe_ratio,
                'max_drawdown': backtest.max_drawdown,
                'config': backtest.config_json
            }
        except Backtest.DoesNotExist:
            return None

    def list_backtests(self, limit: int = 20) -> List[Dict]:
        """
        Get recent backtests

        Args:
            limit: Number of backtests to return

        Returns:
            List of backtest summaries
        """
        backtests = (Backtest
                    .select()
                    .order_by(Backtest.run_date.desc())
                    .limit(limit))

        return [
            {
                'id': b.id,
                'name': b.name,
                'run_date': b.run_date,
                'start_date': b.start_date,
                'end_date': b.end_date,
                'total_trades': b.total_trades,
                'win_rate': b.win_rate,
                'total_pnl': b.total_pnl,
                'sharpe_ratio': b.sharpe_ratio
            }
            for b in backtests
        ]


# Global singleton instance
backtest_service = BacktestService()


if __name__ == '__main__':
    # Test backtesting on November 2025 data (when predictions have actuals)
    from datetime import date

    service = BacktestService()

    try:
        results = service.run_simulation(
            name="Test Nov 2025",
            start_date=date(2025, 11, 1),
            end_date=date(2025, 11, 30),
            confidence_threshold=0.7,
            layer=None  # fusion
        )

        print("\nSample trades:")
        for trade in results['trades'][:5]:
            print(f"  {trade['timestamp']}: {trade['pnl_pct']:+.2f}% (conf: {trade['confidence']:.2f})")

    except Exception as e:
        print(f"Error: {e}")
