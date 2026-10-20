"""Features for a direct multi-week forecast, computed at a forecast origin.

`origin_features(hist, t, h)` cuts the history after week t, i.e. it only sees weeks that are
complete when the order for the following weeks is placed. Demand features are divided by
the article's recent level, so one global model can learn patterns across articles with
very different volumes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import History

N_LAGS = 8


def _mean(Y: np.ndarray, start: int, end: int) -> np.ndarray:
    """Mean over columns start..end (inclusive), NaN if the window is not fully available."""
    if start < 0 or end >= Y.shape[1] or end < start:
        return np.full(Y.shape[0], np.nan)
    return Y[:, start : end + 1].mean(axis=1)


def scale(Y: np.ndarray, t: int) -> np.ndarray:
    """Recent level per article: mean of the last 13 weeks, at least 1."""
    return np.maximum(Y[:, max(0, t - 12) : t + 1].mean(axis=1), 1.0)


def origin_features(hist: History, t: int, h: int) -> pd.DataFrame:
    """Feature rows (one per article) for the forecast of week t+h made after week t."""
    hist = hist.upto(t)
    Y, O = hist.Y, hist.orders
    s = scale(Y, t)
    n = len(s)
    f: dict[str, np.ndarray] = {}
    for lag in range(N_LAGS):
        f[f"lag{lag + 1}"] = (Y[:, t - lag] if t - lag >= 0 else np.full(n, np.nan)) / s
    for w in (4, 8, 26, 52):
        f[f"mean{w}"] = _mean(Y, t - w + 1, t) / s
    recent = Y[:, max(0, t - 12) : t + 1]
    f["median13"] = np.median(recent, axis=1) / s
    f["std13"] = recent.std(axis=1) / s
    f["max13"] = recent.max(axis=1) / s
    f["active13"] = (recent > 0).mean(axis=1)
    sold = Y[:, ::-1] > 0
    f["weeks_since_sale"] = np.where(sold.any(axis=1), np.argmax(sold, axis=1), 52).clip(max=52)

    # Number of order lines is far steadier than quantity, which large orders dominate.
    so = scale(O, t)
    for lag in range(4):
        f[f"orders_lag{lag + 1}"] = (O[:, t - lag] if t - lag >= 0 else np.full(n, np.nan)) / so
    f["orders_mean4"] = _mean(O, t - 3, t) / so
    f["log_orders"] = np.log(so)
    f["qty_per_order13"] = recent.sum(axis=1) / np.maximum(O[:, max(0, t - 12) : t + 1].sum(axis=1), 1)

    # Last year: the target week itself, its neighbourhood, and how demand developed from
    # "now" to the target week one year ago (captures the run-up to Christmas).
    target_ly = t + h - 52
    f["ly_target"] = (Y[:, target_ly] if 0 <= target_ly <= t else np.full(n, np.nan)) / s
    ly_window = _mean(Y, target_ly - 1, target_ly + 1)
    ly_now = _mean(Y, t - 52 - 3, t - 52)
    f["ly_window"] = ly_window / s
    f["ly_growth_to_target"] = ly_window / np.maximum(ly_now, 1.0)
    f["yoy_level"] = _mean(Y, t - 12, t) / np.maximum(_mean(Y, t - 64, t - 52), 1.0)

    f["log_level"] = np.log(s)
    p_now = hist.price[:, t]
    f["log_price"] = np.log(p_now)
    f["price_vs_26w"] = p_now / hist.price[:, max(0, t - 25) : t + 1].mean(axis=1)

    target_week = hist.weeks[t] + pd.Timedelta(weeks=h)
    f["target_weekofyear"] = np.full(n, target_week.isocalendar().week, dtype=float)
    f["horizon"] = np.full(n, h, dtype=float)
    return pd.DataFrame(f)
