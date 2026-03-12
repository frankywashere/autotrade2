"""One-time script to seed evolve_state.json from previous run results."""
import json
import os

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

state = {
    "iteration": 6,
    "best_score": 1144512.0,
    "best_knobs": {
        "exit_grace_bars": 4, "stop_update_secs": 45, "stop_check_secs": 10,
        "grace_ratchet_secs": 225, "profit_activated_stop": False,
        "max_underwater_mins": 90, "max_hold_bars": 60,
        "breakout_stop_mult": 0.55, "eval_interval": 2,
    },
    "best_metrics": {
        "combined_score": 1144512.0, "total_pnl": 45116.0, "n_trades": 5679,
        "win_rate": 59.5, "sharpe": 0.336, "profit_factor": 1.05,
        "max_drawdown_pct": 16.6,
    },
    "history": [
        {"score": 319433, "pnl": 9204, "trades": 2853, "wr": 61.1, "sharpe": 0.116, "dd": 17.2,
         "knobs": {"exit_grace_bars": 5, "stop_update_secs": 60, "stop_check_secs": 5, "grace_ratchet_secs": 300, "profit_activated_stop": True, "max_underwater_mins": 0, "max_hold_bars": 60, "breakout_stop_mult": 1.0, "eval_interval": 3}},
        {"score": 738845, "pnl": 31805, "trades": 5568, "wr": 62.2, "sharpe": 0.247, "dd": 19.4,
         "knobs": {"exit_grace_bars": 3, "stop_update_secs": 30, "stop_check_secs": 10, "grace_ratchet_secs": 150, "profit_activated_stop": False, "max_underwater_mins": 45, "max_hold_bars": 40, "breakout_stop_mult": 0.7, "eval_interval": 2}},
        {"score": 544190, "pnl": 18648, "trades": 5776, "wr": 61.6, "sharpe": 0.17, "dd": 12.5,
         "knobs": {"exit_grace_bars": 2, "stop_update_secs": 15, "stop_check_secs": 5, "grace_ratchet_secs": 90, "profit_activated_stop": False, "max_underwater_mins": 30, "max_hold_bars": 30, "breakout_stop_mult": 0.5, "eval_interval": 2}},
        {"score": 1040323, "pnl": 43016, "trades": 5631, "wr": 60.1, "sharpe": 0.318, "dd": 18.0,
         "knobs": {"exit_grace_bars": 4, "stop_update_secs": 45, "stop_check_secs": 10, "grace_ratchet_secs": 200, "profit_activated_stop": False, "max_underwater_mins": 60, "max_hold_bars": 50, "breakout_stop_mult": 0.6, "eval_interval": 2}},
        {"score": 1053339, "pnl": 42999, "trades": 5653, "wr": 59.8, "sharpe": 0.319, "dd": 17.5,
         "knobs": {"exit_grace_bars": 4, "stop_update_secs": 45, "stop_check_secs": 10, "grace_ratchet_secs": 225, "profit_activated_stop": False, "max_underwater_mins": 75, "max_hold_bars": 55, "breakout_stop_mult": 0.58, "eval_interval": 2}},
        {"score": 1144512, "pnl": 45116, "trades": 5679, "wr": 59.5, "sharpe": 0.336, "dd": 16.6,
         "knobs": {"exit_grace_bars": 4, "stop_update_secs": 45, "stop_check_secs": 10, "grace_ratchet_secs": 225, "profit_activated_stop": False, "max_underwater_mins": 90, "max_hold_bars": 60, "breakout_stop_mult": 0.55, "eval_interval": 2}},
    ],
}

path = os.path.join(OUTPUT_DIR, "evolve_state.json")
with open(path, "w", encoding="utf-8") as f:
    json.dump(state, f, indent=2)
print(f"State seeded at iteration {state['iteration']}, best_score={state['best_score']}")
