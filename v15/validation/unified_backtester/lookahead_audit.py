#!/usr/bin/env python3
"""
Lookahead audit for unified_backtester intraday and surfer_ml algos.

This runs the real backtest engine with instrumented algos and logs:
1. Each actual entry signal the engine emits.
2. Counterfactual signals when the current 5-minute bar is reduced to
   O=H=L=C=open and volume=0.
3. Exit checks compared against the same open-only current bar.

Usage:
    python -m v15.validation.unified_backtester.lookahead_audit --algo intraday
    python -m v15.validation.unified_backtester.lookahead_audit --algo surfer-ml
"""

from __future__ import annotations

import argparse
import copy
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .data_provider import DataProvider
from .engine import BacktestEngine
from .portfolio import PortfolioManager
from .algos.intraday import (
    DEFAULT_INTRADAY_CONFIG,
    IntradayAlgo,
    _channel_position,
    _channel_slope,
    _compute_gap_pct,
    _compute_spread_pct,
    _compute_volume_ratio,
    _compute_vwap,
    _compute_vwap_slope,
    _resample_ohlcv,
)
from .algos.surfer_ml import DEFAULT_SURFER_ML_CONFIG, SurferMLAlgo
from ...trading.intraday_signals import sig_union_enhanced


def _default_data_path() -> Optional[str]:
    for candidate in ["data/TSLAMin.txt", "../data/TSLAMin.txt", "C:/AI/x14/data/TSLAMin.txt"]:
        if os.path.isfile(candidate):
            return candidate
    return None


def _normalize_signal_obj(sig) -> Dict[str, object]:
    if sig is None:
        return {
            "present": False,
            "direction": "",
            "confidence": np.nan,
            "stop_pct": np.nan,
            "tp_pct": np.nan,
            "signal_type": "",
        }
    return {
        "present": True,
        "direction": getattr(sig, "direction", ""),
        "confidence": float(getattr(sig, "confidence", np.nan)),
        "stop_pct": float(getattr(sig, "stop_pct", np.nan)),
        "tp_pct": float(getattr(sig, "tp_pct", np.nan)),
        "signal_type": getattr(sig, "signal_type", ""),
    }


def _normalize_intraday_result(result) -> Dict[str, object]:
    if result is None:
        return {
            "present": False,
            "direction": "",
            "confidence": np.nan,
            "stop_pct": np.nan,
            "tp_pct": np.nan,
            "signal_type": "",
        }
    direction, confidence, stop_pct, tp_pct = result
    return {
        "present": True,
        "direction": direction.lower(),
        "confidence": float(confidence),
        "stop_pct": float(stop_pct),
        "tp_pct": float(tp_pct),
        "signal_type": "intraday",
    }


def _signal_changed(a: Dict[str, object], b: Dict[str, object], tol: float = 1e-12) -> bool:
    if a["present"] != b["present"]:
        return True
    if not a["present"]:
        return False
    if a["direction"] != b["direction"]:
        return True
    if a["signal_type"] != b["signal_type"]:
        return True
    for key in ("confidence", "stop_pct", "tp_pct"):
        av = a[key]
        bv = b[key]
        if (np.isnan(av) and np.isnan(bv)) or abs(av - bv) <= tol:
            continue
        return True
    return False


def _open_only_bar(bar: dict) -> dict:
    out = dict(bar)
    open_px = float(bar["open"])
    out["high"] = open_px
    out["low"] = open_px
    out["close"] = open_px
    out["volume"] = 0.0
    return out


def _prefix_signal_fields(prefix: str, payload: Dict[str, object]) -> Dict[str, object]:
    return {
        f"{prefix}_present": payload["present"],
        f"{prefix}_direction": payload["direction"],
        f"{prefix}_confidence": payload["confidence"],
        f"{prefix}_stop_pct": payload["stop_pct"],
        f"{prefix}_tp_pct": payload["tp_pct"],
        f"{prefix}_signal_type": payload["signal_type"],
    }


def _exit_map(exits: List) -> Dict[str, dict]:
    return {
        ex.pos_id: {"present": True, "price": float(ex.price), "reason": ex.reason}
        for ex in exits
    }


class AuditedIntradayAlgo(IntradayAlgo):
    def __init__(self, config=None, data=None):
        super().__init__(config or copy.deepcopy(DEFAULT_INTRADAY_CONFIG), data)
        self.entry_audit: List[dict] = []
        self.exit_audit: List[dict] = []
        self._wins = {"5m": 60, "1h": 24, "4h": 20}
        self._build_completed_htf_lookup()

    def _build_completed_htf_lookup(self):
        f5m = self._f5m
        n = len(f5m)
        self._h1_cp_completed = np.full(n, np.nan)
        self._h1_slope_completed = np.full(n, np.nan)
        self._h4_cp_completed = np.full(n, np.nan)
        self._h4_slope_completed = np.full(n, np.nan)

        df1m = self.data._tf_data["1min"]
        specs = [
            ("1h", pd.Timedelta(hours=1), self._h1_cp_completed, self._h1_slope_completed),
            ("4h", pd.Timedelta(hours=4), self._h4_cp_completed, self._h4_slope_completed),
        ]
        for rule, period, out_cp, out_slope in specs:
            tf = _resample_ohlcv(df1m, rule)
            c_tf = tf["close"].values.astype(np.float64)
            tf_feat = pd.DataFrame(index=tf.index)
            tf_feat["chan_pos"] = _channel_position(c_tf, self._wins[rule])
            tf_feat["chan_slope"] = _channel_slope(c_tf, self._wins[rule])
            ti = pd.DatetimeIndex(tf_feat.index)
            cp_vals = tf_feat["chan_pos"].values
            slope_vals = tf_feat["chan_slope"].values
            j = 0
            for i, t in enumerate(f5m.index):
                while j < len(ti) - 1 and ti[j + 1] + period <= t:
                    j += 1
                if j < len(ti) and ti[j] + period <= t:
                    out_cp[i] = cp_vals[j]
                    out_slope[i] = slope_vals[j]

    def _intraday_features(self, idx: int, mode: str = "actual", use_completed_htf: bool = False) -> Dict[str, float]:
        o = self._f5m["open"].values.astype(np.float64).copy()
        h = self._f5m["high"].values.astype(np.float64).copy()
        l = self._f5m["low"].values.astype(np.float64).copy()
        c = self._f5m["close"].values.astype(np.float64).copy()
        v = self._f5m["volume"].values.astype(np.float64).copy()
        dates = np.array([t.date() for t in self._f5m.index])

        if mode in ("close_only", "open_only"):
            c[idx] = o[idx]
        if mode in ("range_only", "open_only"):
            h[idx] = o[idx]
            l[idx] = o[idx]
        if mode in ("volume_only", "open_only"):
            v[idx] = 0.0

        cp5_arr = _channel_position(c, self._wins["5m"])
        _, vwap_dist_arr = _compute_vwap(o, h, l, c, v, dates)
        vol_ratio_arr = _compute_volume_ratio(v)
        vwap_slope_arr = _compute_vwap_slope(vwap_dist_arr)
        spread_pct_arr = _compute_spread_pct(h, l, c)
        gap_pct_arr = _compute_gap_pct(c, dates)

        return {
            "cp5": cp5_arr[idx],
            "vwap_dist": vwap_dist_arr[idx],
            "daily_cp": self._daily_cp[idx],
            "h1_cp": self._h1_cp_completed[idx] if use_completed_htf else self._h1_cp[idx],
            "h4_cp": self._h4_cp_completed[idx] if use_completed_htf else self._h4_cp[idx],
            "vol_ratio": vol_ratio_arr[idx],
            "vwap_slope": vwap_slope_arr[idx],
            "gap_pct": gap_pct_arr[idx],
            "spread_pct": spread_pct_arr[idx],
            "daily_slope": self._daily_slope[idx],
            "h1_slope": self._h1_slope_completed[idx] if use_completed_htf else self._h1_slope[idx],
            "h4_slope": self._h4_slope_completed[idx] if use_completed_htf else self._h4_slope[idx],
        }

    def _signal_from_features(self, feat: Dict[str, float]):
        sig_params = self.config.params.get("signal_params", {}) or None
        return sig_union_enhanced(
            cp5=feat["cp5"],
            vwap_dist=feat["vwap_dist"],
            daily_cp=feat["daily_cp"],
            h1_cp=feat["h1_cp"],
            h4_cp=feat["h4_cp"],
            vol_ratio=feat["vol_ratio"],
            vwap_slope=feat["vwap_slope"],
            gap_pct=feat["gap_pct"],
            daily_slope=feat["daily_slope"],
            h1_slope=feat["h1_slope"],
            h4_slope=feat["h4_slope"],
            spread_pct=feat["spread_pct"],
            params=sig_params,
        )

    def on_bar(self, time: pd.Timestamp, bar: dict, open_positions: list):
        signals = super().on_bar(time, bar, open_positions)
        if not signals:
            return signals

        idx = self._bar_index[time]
        actual = _normalize_signal_obj(signals[0])
        open_only = _normalize_intraday_result(self._signal_from_features(self._intraday_features(idx, "open_only")))
        close_only = _normalize_intraday_result(self._signal_from_features(self._intraday_features(idx, "close_only")))
        range_only = _normalize_intraday_result(self._signal_from_features(self._intraday_features(idx, "range_only")))
        volume_only = _normalize_intraday_result(self._signal_from_features(self._intraday_features(idx, "volume_only")))
        completed_htf = _normalize_intraday_result(
            self._signal_from_features(self._intraday_features(idx, "actual", use_completed_htf=True))
        )
        corrected = _normalize_intraday_result(
            self._signal_from_features(self._intraday_features(idx, "open_only", use_completed_htf=True))
        )

        row = {
            "algo": self.algo_id,
            "time": time,
            "bar_open": float(bar["open"]),
            "bar_high": float(bar["high"]),
            "bar_low": float(bar["low"]),
            "bar_close": float(bar["close"]),
            "bar_volume": float(bar["volume"]),
            "h1_cp_engine": float(self._h1_cp[idx]) if not np.isnan(self._h1_cp[idx]) else np.nan,
            "h1_cp_completed": float(self._h1_cp_completed[idx]) if not np.isnan(self._h1_cp_completed[idx]) else np.nan,
            "h4_cp_engine": float(self._h4_cp[idx]) if not np.isnan(self._h4_cp[idx]) else np.nan,
            "h4_cp_completed": float(self._h4_cp_completed[idx]) if not np.isnan(self._h4_cp_completed[idx]) else np.nan,
            "h1_slope_engine": float(self._h1_slope[idx]) if not np.isnan(self._h1_slope[idx]) else np.nan,
            "h1_slope_completed": float(self._h1_slope_completed[idx]) if not np.isnan(self._h1_slope_completed[idx]) else np.nan,
            "h4_slope_engine": float(self._h4_slope[idx]) if not np.isnan(self._h4_slope[idx]) else np.nan,
            "h4_slope_completed": float(self._h4_slope_completed[idx]) if not np.isnan(self._h4_slope_completed[idx]) else np.nan,
            "uses_current_close": _signal_changed(actual, close_only),
            "uses_current_high_low": _signal_changed(actual, range_only),
            "uses_current_volume": _signal_changed(actual, volume_only),
            "changed_open_only": _signal_changed(actual, open_only),
            "changed_completed_htf": _signal_changed(actual, completed_htf),
            "changed_corrected": _signal_changed(actual, corrected),
        }
        row.update(_prefix_signal_fields("actual", actual))
        row.update(_prefix_signal_fields("open_only", open_only))
        row.update(_prefix_signal_fields("close_only", close_only))
        row.update(_prefix_signal_fields("range_only", range_only))
        row.update(_prefix_signal_fields("volume_only", volume_only))
        row.update(_prefix_signal_fields("completed_htf", completed_htf))
        row.update(_prefix_signal_fields("corrected", corrected))
        self.entry_audit.append(row)
        return signals

    def check_exits(self, time: pd.Timestamp, bar: dict, open_positions: list):
        baseline = super().check_exits(time, bar, open_positions)
        counterfactual = super().check_exits(time, _open_only_bar(bar), open_positions)
        baseline_map = _exit_map(baseline)
        counter_map = _exit_map(counterfactual)

        for pos in open_positions:
            base = baseline_map.get(pos.pos_id, {"present": False, "price": np.nan, "reason": ""})
            cf = counter_map.get(pos.pos_id, {"present": False, "price": np.nan, "reason": ""})
            self.exit_audit.append(
                {
                    "algo": self.algo_id,
                    "time": time,
                    "pos_id": pos.pos_id,
                    "entry_time": pos.entry_time,
                    "same_bar_as_fill": pos.entry_time == time,
                    "bar_open": float(bar["open"]),
                    "bar_high": float(bar["high"]),
                    "bar_low": float(bar["low"]),
                    "bar_close": float(bar["close"]),
                    "bar_volume": float(bar["volume"]),
                    "baseline_exit": base["present"],
                    "baseline_exit_reason": base["reason"],
                    "baseline_exit_price": base["price"],
                    "open_only_exit": cf["present"],
                    "open_only_exit_reason": cf["reason"],
                    "open_only_exit_price": cf["price"],
                    "changed": base != cf,
                }
            )
        return baseline


class AuditedSurferMLAlgo(SurferMLAlgo):
    def __init__(self, config=None, data=None):
        super().__init__(config or copy.deepcopy(DEFAULT_SURFER_ML_CONFIG), data)
        self.entry_audit: List[dict] = []
        self.exit_audit: List[dict] = []

    @staticmethod
    def _atr_from_df(bars: pd.DataFrame, period: int) -> float:
        if len(bars) < period + 1:
            return 0.0
        recent = bars.tail(period + 1)
        highs = recent["high"].values
        lows = recent["low"].values
        closes = recent["close"].values
        tr = np.maximum(
            highs[1:] - lows[1:],
            np.maximum(np.abs(highs[1:] - closes[:-1]), np.abs(lows[1:] - closes[:-1])),
        )
        return float(np.mean(tr))

    @staticmethod
    def _apply_mode_to_current_bar(df5: pd.DataFrame, time: pd.Timestamp, mode: str) -> pd.DataFrame:
        if mode == "actual" or len(df5) == 0 or df5.index[-1] != time:
            return df5
        out = df5.copy()
        open_px = float(out.iloc[-1]["open"])
        if mode in ("close_only", "open_only"):
            out.iloc[-1, out.columns.get_loc("close")] = open_px
        if mode in ("range_only", "open_only"):
            out.iloc[-1, out.columns.get_loc("high")] = open_px
            out.iloc[-1, out.columns.get_loc("low")] = open_px
        if mode in ("volume_only", "open_only") and "volume" in out.columns:
            out.iloc[-1, out.columns.get_loc("volume")] = 0.0
        return out

    def _surfer_signal_snapshot(self, time: pd.Timestamp, bar: dict, open_positions: list, mode: str = "actual"):
        try:
            from v15.core.channel import detect_channels_multi_window, select_best_channel
            from v15.core.channel_surfer import analyze_channels, TF_WINDOWS
            from .algo_base import Signal
        except ImportError:
            return None

        existing_dirs = {p.direction for p in open_positions}
        existing_types = {getattr(p, "signal_type", "") for p in open_positions}

        df5 = self.data.get_bars("5min", time)
        if len(df5) < 20:
            return None
        df5 = self._apply_mode_to_current_bar(df5, time, mode)
        df_slice = df5.tail(100)

        try:
            multi = detect_channels_multi_window(df_slice, windows=[10, 15, 20, 30, 40])
            best_ch, _ = select_best_channel(multi)
        except Exception:
            return None

        if best_ch is None or not best_ch.valid:
            return None

        slice_closes = df_slice["close"].values
        channels_by_tf = {"5min": best_ch}
        prices_by_tf = {"5min": slice_closes}
        current_prices = {"5min": float(slice_closes[-1])}
        volumes_dict = {}
        if "volume" in df_slice.columns:
            volumes_dict["5min"] = df_slice["volume"].values

        tf_periods = {
            "1h": pd.Timedelta(hours=1),
            "4h": pd.Timedelta(hours=4),
            "daily": pd.Timedelta(days=1),
        }
        for tf_label in ("1h", "4h", "daily"):
            try:
                tf_df = self.data.get_bars(tf_label, time)
            except (ValueError, KeyError):
                continue
            if len(tf_df) == 0:
                continue
            tf_available = tf_df[tf_df.index + tf_periods[tf_label] <= time]
            tf_recent = tf_available.tail(100)
            if len(tf_recent) < 30:
                continue
            tf_windows = TF_WINDOWS.get(tf_label, [20, 30, 40])
            try:
                tf_multi = detect_channels_multi_window(tf_recent, windows=tf_windows)
                tf_ch, _ = select_best_channel(tf_multi)
                if tf_ch and tf_ch.valid:
                    channels_by_tf[tf_label] = tf_ch
                    prices_by_tf[tf_label] = tf_recent["close"].values
                    current_prices[tf_label] = float(tf_recent["close"].iloc[-1])
                    if "volume" in tf_recent.columns:
                        volumes_dict[tf_label] = tf_recent["volume"].values
            except Exception:
                continue

        try:
            analysis = analyze_channels(
                channels_by_tf,
                prices_by_tf,
                current_prices,
                volumes_by_tf=volumes_dict if volumes_dict else None,
            )
        except Exception:
            return None

        sig = analysis.signal
        if sig.action not in ("BUY", "SELL"):
            return None

        conf = sig.confidence
        if conf < self.config.params.get("min_confidence", 0.01):
            return None

        direction = "long" if sig.action == "BUY" else "short"
        if direction in existing_dirs:
            return None
        signal_type = sig.signal_type or "bounce"
        if signal_type in existing_types:
            return None

        stop_pct = sig.suggested_stop_pct or self.config.params["stop_pct"]
        tp_pct = sig.suggested_tp_pct or self.config.params["tp_pct"]

        atr_period = self.config.params.get("atr_period", 14)
        atr_val = self._atr_from_df(df5, atr_period)
        if mode == "actual":
            bar_eval = bar
        else:
            bar_eval = _open_only_bar(bar) if mode == "open_only" else dict(bar)
            open_px = float(bar["open"])
            if mode in ("close_only", "open_only"):
                bar_eval["close"] = open_px
            if mode in ("range_only", "open_only"):
                bar_eval["high"] = open_px
                bar_eval["low"] = open_px
            if mode in ("volume_only", "open_only"):
                bar_eval["volume"] = 0.0

        entry_price = float(bar_eval["close"])
        if atr_val > 0 and entry_price > 0:
            if signal_type == "bounce":
                atr_floor = (0.5 * atr_val) / entry_price
                atr_cap = (1.5 * atr_val) / entry_price
            else:
                atr_floor = (1.5 * atr_val) / entry_price
                atr_cap = (3.0 * atr_val) / entry_price
            stop_pct = float(np.clip(stop_pct, atr_floor, atr_cap))

        if signal_type == "break":
            stop_pct *= self.config.params.get("breakout_stop_mult", 1.00)
        if signal_type == "bounce" and conf > 0.65:
            tp_pct *= 1.30

        return Signal(
            algo_id=self.config.algo_id,
            direction=direction,
            price=float(bar_eval["close"]),
            confidence=conf,
            stop_pct=stop_pct,
            tp_pct=tp_pct,
            signal_type=signal_type,
            metadata={
                "el_flagged": False,
                "trail_width_mult": 1.0,
                "fast_reversion": False,
                "ou_half_life": self.config.params.get("ou_half_life", 5.0),
                "signal_bar_high": float(bar_eval["high"]),
                "signal_bar_low": float(bar_eval["low"]),
            },
        )

    def on_bar(self, time: pd.Timestamp, bar: dict, open_positions: list):
        signals = super().on_bar(time, bar, open_positions)
        if not signals:
            return signals

        actual = _normalize_signal_obj(signals[0])
        open_only = _normalize_signal_obj(self._surfer_signal_snapshot(time, bar, open_positions, "open_only"))
        close_only = _normalize_signal_obj(self._surfer_signal_snapshot(time, bar, open_positions, "close_only"))
        range_only = _normalize_signal_obj(self._surfer_signal_snapshot(time, bar, open_positions, "range_only"))
        volume_only = _normalize_signal_obj(self._surfer_signal_snapshot(time, bar, open_positions, "volume_only"))

        row = {
            "algo": self.algo_id,
            "time": time,
            "bar_open": float(bar["open"]),
            "bar_high": float(bar["high"]),
            "bar_low": float(bar["low"]),
            "bar_close": float(bar["close"]),
            "bar_volume": float(bar["volume"]),
            "uses_current_close": _signal_changed(actual, close_only),
            "uses_current_high_low": _signal_changed(actual, range_only),
            "uses_current_volume": _signal_changed(actual, volume_only),
            "changed_open_only": _signal_changed(actual, open_only),
        }
        row.update(_prefix_signal_fields("actual", actual))
        row.update(_prefix_signal_fields("open_only", open_only))
        row.update(_prefix_signal_fields("close_only", close_only))
        row.update(_prefix_signal_fields("range_only", range_only))
        row.update(_prefix_signal_fields("volume_only", volume_only))
        self.entry_audit.append(row)
        return signals

    def check_exits(self, time: pd.Timestamp, bar: dict, open_positions: list):
        state_before = copy.deepcopy(self._pos_state)
        baseline = super().check_exits(time, bar, open_positions)
        state_after = copy.deepcopy(self._pos_state)
        self._pos_state = copy.deepcopy(state_before)
        counterfactual = super().check_exits(time, _open_only_bar(bar), open_positions)
        self._pos_state = state_after

        baseline_map = _exit_map(baseline)
        counter_map = _exit_map(counterfactual)
        for pos in open_positions:
            base = baseline_map.get(pos.pos_id, {"present": False, "price": np.nan, "reason": ""})
            cf = counter_map.get(pos.pos_id, {"present": False, "price": np.nan, "reason": ""})
            self.exit_audit.append(
                {
                    "algo": self.algo_id,
                    "time": time,
                    "pos_id": pos.pos_id,
                    "entry_time": pos.entry_time,
                    "same_bar_as_fill": pos.entry_time == time,
                    "bar_open": float(bar["open"]),
                    "bar_high": float(bar["high"]),
                    "bar_low": float(bar["low"]),
                    "bar_close": float(bar["close"]),
                    "bar_volume": float(bar["volume"]),
                    "baseline_exit": base["present"],
                    "baseline_exit_reason": base["reason"],
                    "baseline_exit_price": base["price"],
                    "open_only_exit": cf["present"],
                    "open_only_exit_reason": cf["reason"],
                    "open_only_exit_price": cf["price"],
                    "changed": base != cf,
                }
            )
        return baseline


def _build_algo(algo_name: str, data: DataProvider):
    if algo_name == "intraday":
        return AuditedIntradayAlgo(copy.deepcopy(DEFAULT_INTRADAY_CONFIG), data)
    if algo_name in ("surfer-ml", "surfer_ml", "surfer"):
        return AuditedSurferMLAlgo(copy.deepcopy(DEFAULT_SURFER_ML_CONFIG), data)
    raise ValueError(f"Unsupported algo '{algo_name}'. Use intraday or surfer-ml.")


def _print_summary(algo_name: str, entry_df: pd.DataFrame, exit_df: pd.DataFrame):
    print(f"\n=== {algo_name} ===")
    if len(entry_df) == 0:
        print("Entry audit: 0 actual signals")
    else:
        print(f"Entry audit: {len(entry_df)} actual signals")
        print(f"  Changed with open-only current 5m bar: {int(entry_df['changed_open_only'].sum())}")
        print(f"  Depends on current close: {int(entry_df['uses_current_close'].sum())}")
        print(f"  Depends on current high/low: {int(entry_df['uses_current_high_low'].sum())}")
        print(f"  Depends on current volume: {int(entry_df['uses_current_volume'].sum())}")
        if "changed_completed_htf" in entry_df.columns:
            print(f"  Changed with completed-only 1h/4h mapping: {int(entry_df['changed_completed_htf'].sum())}")
            print(f"  Changed with both fixes applied: {int(entry_df['changed_corrected'].sum())}")

    if len(exit_df) == 0:
        print("Exit audit: 0 exit checks")
    else:
        print(f"Exit audit: {len(exit_df)} position-level exit checks")
        print(f"  Changed with open-only current 5m bar: {int(exit_df['changed'].sum())}")
        print(f"  Same timestamp as fill: {int(exit_df['same_bar_as_fill'].sum())}")
        print(f"  Baseline exits fired: {int(exit_df['baseline_exit'].sum())}")


def _write_csv(prefix: str, algo_name: str, entry_df: pd.DataFrame, exit_df: pd.DataFrame):
    base = f"{prefix}_{algo_name.replace('-', '_')}"
    entry_path = f"{base}_entries.csv"
    exit_path = f"{base}_exits.csv"
    entry_df.to_csv(entry_path, index=False)
    exit_df.to_csv(exit_path, index=False)
    print(f"  Wrote {entry_path}")
    print(f"  Wrote {exit_path}")


def main():
    parser = argparse.ArgumentParser(description="Lookahead audit for unified_backtester algos")
    parser.add_argument("--data", type=str, default=None, help="Path to TSLAMin.txt")
    parser.add_argument("--spy", type=str, default=None, help="Path to SPYMin.txt")
    parser.add_argument("--start", type=str, default="2024-01-01")
    parser.add_argument("--end", type=str, default="2025-03-01")
    parser.add_argument("--algo", type=str, action="append", default=[], help="intraday or surfer-ml")
    parser.add_argument("--csv-prefix", type=str, default=None, help="Write CSVs to <prefix>_<algo>_entries.csv/exits.csv")
    parser.add_argument("--quiet", action="store_true", help="Suppress engine progress output")
    args = parser.parse_args()

    if args.data is None:
        args.data = _default_data_path()
    if args.data is None:
        raise SystemExit("ERROR: Could not find TSLAMin.txt. Use --data.")

    algos = args.algo or ["intraday", "surfer-ml"]
    print(f"Loading data from {args.data}...")
    data = DataProvider(
        tsla_1min_path=args.data,
        start=args.start,
        end=args.end,
        spy_path=args.spy,
        rth_only=True,
    )
    print(f"Loaded {len(data._df1m):,} 1-min bars")

    for algo_name in algos:
        try:
            algo = _build_algo(algo_name, data)
        except IndexError as exc:
            raise SystemExit(
                f"ERROR: Failed to initialize {algo_name}. "
                "This audit needs enough warmup history for higher-timeframe feature windows. "
                "Try an earlier --start date."
            ) from exc
        portfolio = PortfolioManager()
        portfolio.register_algo(
            algo_id=algo.config.algo_id,
            initial_equity=algo.config.initial_equity,
            max_per_trade=algo.config.max_equity_per_trade,
            max_positions=algo.config.max_positions,
            cost_model=algo.config.cost_model,
        )
        engine = BacktestEngine(data, [algo], portfolio, verbose=not args.quiet)
        engine.run()

        entry_df = pd.DataFrame(algo.entry_audit)
        exit_df = pd.DataFrame(algo.exit_audit)
        _print_summary(algo_name, entry_df, exit_df)
        if args.csv_prefix:
            _write_csv(args.csv_prefix, algo_name, entry_df, exit_df)


if __name__ == "__main__":
    main()
