"""
Peewee ORM models for predictions.db

Maps to existing predictions table and adds new tables for:
- trades
- backtests
- alerts
- users
"""
from peewee import *
from pathlib import Path
from datetime import datetime

# Database connection to existing predictions.db
db_path = Path(__file__).parent.parent.parent.parent / 'data' / 'predictions.db'
db = SqliteDatabase(str(db_path))


class BaseModel(Model):
    """Base model with database connection"""
    class Meta:
        database = db


class Prediction(BaseModel):
    """
    Maps to EXISTING predictions table (do not create)

    Schema has 45+ columns including:
    - predicted_high, predicted_low, confidence
    - actual_high, actual_low, has_actuals
    - fast/medium/slow layer predictions
    - multi-task outputs (hit_band, expected_return, etc.)
    """
    id = AutoField()
    timestamp = DateTimeField(default=datetime.now)
    symbol = CharField(default='TSLA')
    current_price = FloatField()

    # Primary predictions
    predicted_high = FloatField()
    predicted_low = FloatField()
    confidence = FloatField()

    # Actuals (for validation)
    actual_high = FloatField(null=True)
    actual_low = FloatField(null=True)
    has_actuals = BooleanField(default=False)

    # Sub-predictions (from existing schema)
    sub_pred_15min_high = FloatField(null=True)
    sub_pred_15min_low = FloatField(null=True)
    sub_pred_15min_conf = FloatField(null=True)

    sub_pred_1hour_high = FloatField(null=True)
    sub_pred_1hour_low = FloatField(null=True)
    sub_pred_1hour_conf = FloatField(null=True)

    sub_pred_4hour_high = FloatField(null=True)
    sub_pred_4hour_low = FloatField(null=True)
    sub_pred_4hour_conf = FloatField(null=True)

    sub_pred_daily_high = FloatField(null=True)
    sub_pred_daily_low = FloatField(null=True)
    sub_pred_daily_conf = FloatField(null=True)

    # Additional fields from existing schema
    predicted_center = FloatField(null=True)
    predicted_range = FloatField(null=True)
    error_high = FloatField(null=True)
    error_low = FloatField(null=True)
    error_center = FloatField(null=True)
    absolute_error = FloatField(null=True)

    channel_position = FloatField(null=True)
    rsi_value = FloatField(null=True)
    spy_correlation = FloatField(null=True)

    has_earnings = BooleanField(null=True)
    has_macro_event = BooleanField(null=True)
    event_type = CharField(null=True)

    model_version = CharField(null=True)
    feature_dim = IntegerField(null=True)
    timeframe = CharField(null=True)
    model_timeframe = CharField(null=True)
    is_ensemble = BooleanField(null=True)
    news_enabled = BooleanField(null=True)
    prediction_timestamp = DateTimeField(null=True)
    target_timestamp = DateTimeField(null=True)
    simulation_date = DateTimeField(null=True)

    class Meta:
        table_name = 'predictions'


class Trade(BaseModel):
    """Manual trade tracking"""
    id = AutoField()
    prediction = ForeignKeyField(Prediction, backref='trades', null=True)

    entry_time = DateTimeField()
    entry_price = FloatField()

    exit_time = DateTimeField(null=True)
    exit_price = FloatField(null=True)

    pnl = FloatField(null=True)
    pnl_pct = FloatField(null=True)

    quantity = IntegerField(default=1)
    notes = TextField(null=True)

    class Meta:
        table_name = 'trades'


class Backtest(BaseModel):
    """Backtest run results"""
    id = AutoField()
    name = CharField()
    run_date = DateTimeField(default=datetime.now)

    start_date = DateField()
    end_date = DateField()

    total_trades = IntegerField(null=True)
    win_rate = FloatField(null=True)
    total_pnl = FloatField(null=True)
    sharpe_ratio = FloatField(null=True)
    max_drawdown = FloatField(null=True)

    config_json = TextField(null=True)  # JSON string with configuration

    class Meta:
        table_name = 'backtests'


class Alert(BaseModel):
    """Alert configuration and history"""
    id = AutoField()
    type = CharField()  # 'high_confidence', 'channel_break', etc.
    condition_json = TextField()  # JSON string with conditions

    triggered_at = DateTimeField(null=True)
    sent_via = CharField(null=True)  # 'email', 'telegram', 'both'

    prediction = ForeignKeyField(Prediction, backref='alerts', null=True)
    dismissed = BooleanField(default=False)

    class Meta:
        table_name = 'alerts'


class User(BaseModel):
    """User accounts (for authentication)"""
    id = AutoField()
    username = CharField(unique=True)
    email = CharField(unique=True)
    password_hash = CharField()

    created_at = DateTimeField(default=datetime.now)

    class Meta:
        table_name = 'users'


def init_db():
    """
    Initialize database - create new tables only

    IMPORTANT: Does NOT touch existing predictions table
    """
    db.connect()

    # Create only NEW tables (predictions table already exists)
    db.create_tables([Trade, Backtest, Alert, User], safe=True)

    print("✓ Database initialized")
    print(f"  Location: {db_path}")
    print(f"  New tables: trades, backtests, alerts, users")


def get_db():
    """Get database connection (for FastAPI dependency injection)"""
    if db.is_closed():
        db.connect()
    try:
        yield db
    finally:
        if not db.is_closed():
            db.close()


if __name__ == '__main__':
    # Test database connection
    init_db()

    # Verify predictions table exists
    try:
        count = Prediction.select().count()
        print(f"✓ Found {count} existing predictions")
    except Exception as e:
        print(f"✗ Error reading predictions: {e}")
