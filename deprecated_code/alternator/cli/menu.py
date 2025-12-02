from alternator.config import load_app_config
from alternator.data import download_price_data
from alternator.db import init_databases
from alternator.training import train_model, backtest_model


def _prompt(prompt: str, default: str | None = None) -> str:
    if default is not None:
        full = f"{prompt} [{default}]: "
    else:
        full = f"{prompt}: "
    val = input(full).strip()
    return val or (default or "")


def run_cli_menu() -> None:
    app_config = load_app_config()
    init_databases(app_config.paths.db_dir)

    while True:
        print("\n=== Alternator Main Menu ===")
        print("1) Download data (yfinance)")
        print("2) Train hierarchical model")
        print("3) Backtest model")
        print("4) Exit")
        choice = _prompt("Select an option", "4")

        if choice == "1":
            symbol = _prompt("Symbol", app_config.training.symbol)
            start = _prompt("Start date (YYYY-MM-DD)", "2015-01-01")
            end = _prompt("End date (YYYY-MM-DD)", "2024-12-31")
            interval = _prompt("Interval (1m, 5m, 15m, 1h, 1d)", "1m")
            try:
                path = download_price_data(
                    symbol=symbol,
                    start=start,
                    end=end,
                    interval=interval,
                    out_dir=app_config.paths.data_dir,
                )
                print(f"Saved data to {path}")
            except Exception as e:
                print(f"Error downloading data: {e}")

        elif choice == "2":
            csv_name = _prompt(
                "CSV filename in data/ (e.g., TSLA_1m_2015-01-01_2024-12-31.csv)",
                "",
            )
            csv_path = app_config.paths.data_dir / csv_name
            if not csv_path.exists():
                print(f"CSV not found: {csv_path}")
                continue
            epochs = _prompt("Epochs", str(app_config.training.epochs))
            try:
                app_config.training.epochs = int(epochs)
            except ValueError:
                print("Invalid epochs; using default.")
            train_model(app_config, csv_path)

        elif choice == "3":
            csv_name = _prompt(
                "CSV filename in data/ for backtest",
                "",
            )
            csv_path = app_config.paths.data_dir / csv_name
            if not csv_path.exists():
                print(f"CSV not found: {csv_path}")
                continue
            model_name = _prompt(
                "Model filename in models/ (e.g., hierarchical_model.pth)",
                "hierarchical_model.pth",
            )
            model_path = app_config.paths.models_dir / model_name
            if not model_path.exists():
                print(f"Model file not found: {model_path}")
                continue
            backtest_model(app_config, csv_path, model_path)
            print("Backtest completed and logged to DB.")

        elif choice == "4":
            print("Exiting Alternator CLI.")
            break
        else:
            print("Invalid choice. Please try again.")


