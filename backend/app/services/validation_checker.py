"""
Validation Checker Service

Background service that:
1. Runs hourly to check predictions that need validation
2. Fetches actual prices from market data
3. Triggers online learning updates if errors are high
4. Tracks model adaptation over time

This is the CRITICAL missing piece for LNN online learning!
"""
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime, timedelta
from pathlib import Path
import sys
import yfinance as yf

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from backend.app.models.database import Prediction


class ValidationChecker:
    """
    Background service for validating predictions and triggering online updates
    """

    def __init__(self, online_learner=None):
        """
        Initialize validation checker

        Args:
            online_learner: OnlineLearner instance (optional, can be set later)
        """
        self.online_learner = online_learner
        self.scheduler = BackgroundScheduler()
        self.validation_horizon_hours = 24  # Check predictions 24 hours after they were made

        print("✓ ValidationChecker initialized")

    def check_pending_validations(self):
        """
        Check all predictions that need validation

        Called hourly by scheduler
        """
        print(f"\n[{datetime.now()}] Running validation check...")

        # Find predictions that:
        # 1. Don't have actuals yet (has_actuals=False)
        # 2. Were made more than validation_horizon_hours ago
        cutoff_time = datetime.now() - timedelta(hours=self.validation_horizon_hours)

        pending = (Prediction
                  .select()
                  .where(
                      (Prediction.has_actuals == False) &
                      (Prediction.timestamp <= cutoff_time)
                  )
                  .order_by(Prediction.timestamp)
                  .limit(100))  # Process max 100 per hour

        pending_list = list(pending)
        print(f"  Found {len(pending_list)} predictions needing validation")

        if len(pending_list) == 0:
            return {'validated': 0, 'updated': 0}

        validated_count = 0
        updated_count = 0

        for pred in pending_list:
            try:
                # Fetch actual prices for the period after prediction
                actual_high, actual_low = self._fetch_actuals(
                    symbol=pred.symbol or 'TSLA',
                    timestamp=pred.timestamp,
                    horizon_hours=self.validation_horizon_hours
                )

                if actual_high is None or actual_low is None:
                    print(f"  ⚠️ Could not fetch actuals for prediction #{pred.id}")
                    continue

                # Update prediction with actuals
                pred.actual_high = actual_high
                pred.actual_low = actual_low
                pred.has_actuals = True

                # Calculate errors
                pred.error_high = pred.predicted_high - actual_high
                pred.error_low = pred.predicted_low - actual_low
                pred.absolute_error = (abs(pred.error_high) + abs(pred.error_low)) / 2

                pred.save()

                validated_count += 1

                print(f"  ✓ Validated prediction #{pred.id}: MAE={pred.absolute_error:.2f}%")

                # Trigger online learning update if error is high and learner is available
                if self.online_learner and pred.absolute_error >= self.online_learner.error_threshold:
                    result = self.online_learner.check_and_update(pred.id)
                    if result.get('updated'):
                        updated_count += 1

            except Exception as e:
                print(f"  ✗ Error validating prediction #{pred.id}: {e}")
                continue

        print(f"  Summary: {validated_count} validated, {updated_count} triggered updates")

        return {
            'validated': validated_count,
            'updated': updated_count,
            'timestamp': datetime.now().isoformat()
        }

    def _fetch_actuals(
        self,
        symbol: str,
        timestamp: datetime,
        horizon_hours: int = 24
    ) -> tuple:
        """
        Fetch actual high/low prices for validation period

        Args:
            symbol: Stock symbol (TSLA)
            timestamp: Prediction timestamp
            horizon_hours: How many hours forward to check

        Returns:
            (actual_high_pct, actual_low_pct) or (None, None) if unavailable
        """
        try:
            # Get price at prediction time and after validation period
            start_time = timestamp
            end_time = timestamp + timedelta(hours=horizon_hours)

            # Fetch 1-minute data for precise high/low
            ticker = yf.Ticker(symbol)
            df = ticker.history(
                start=start_time,
                end=end_time,
                interval='1m'
            )

            if len(df) == 0:
                return None, None

            # Get price at prediction time (first bar)
            entry_price = df.iloc[0]['Close']

            # Get highest and lowest prices during validation period
            actual_high_price = df['High'].max()
            actual_low_price = df['Low'].min()

            # Convert to percentage moves
            actual_high_pct = ((actual_high_price / entry_price) - 1) * 100
            actual_low_pct = ((actual_low_price / entry_price) - 1) * 100

            return actual_high_pct, actual_low_pct

        except Exception as e:
            print(f"    Error fetching actuals: {e}")
            return None, None

    def start(self):
        """Start background scheduler (runs hourly validation checks)"""
        # Schedule validation checks every hour
        self.scheduler.add_job(
            self.check_pending_validations,
            'interval',
            hours=1,
            id='validation_check'
        )

        self.scheduler.start()
        print("✓ ValidationChecker started (runs every hour)")

    def stop(self):
        """Stop background scheduler"""
        self.scheduler.shutdown()
        print("✓ ValidationChecker stopped")

    def run_manual_check(self):
        """Manually trigger validation check (useful for testing)"""
        return self.check_pending_validations()


# Global instance
validation_checker = None


def init_validation_checker(online_learner=None):
    """Initialize global validation checker"""
    global validation_checker
    validation_checker = ValidationChecker(online_learner)
    return validation_checker


if __name__ == '__main__':
    # Test validation checker
    checker = ValidationChecker()

    print("\nRunning manual validation check...")
    result = checker.run_manual_check()

    print(f"\nResult: {result}")
