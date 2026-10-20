"""Forecasting methods. Each one maps the history up to week t to forecasts for t+1..t+H.

Interface: `fit(hist, t)` and `predict(hist, t, H)` receive a history that ends with week t;
`predict` returns an array (articles × H). The baselines are what planning in Excel or an ERP
usually does.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd

from .data import History
from .features import origin_features, scale

MIN_HISTORY = 13  # first origin used for training


class MovingAverage:
    """Mean of the last `window` weeks, the same for every horizon — the Excel classic."""

    def __init__(self, window: int = 8):
        self.window = window
        self.name = f"Gleitender Durchschnitt ({window} Wochen)"

    def fit(self, hist: History, t: int) -> MovingAverage:
        return self

    def predict(self, hist: History, t: int, H: int) -> np.ndarray:
        Y = hist.Y
        level = Y[:, max(0, t - self.window + 1) : t + 1].mean(axis=1)
        return np.repeat(level[:, None], H, axis=1)


class ExponentialSmoothing:
    """Simple exponential smoothing — the default forecast in many ERP systems."""

    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha
        self.name = f"Exponentielle Glättung (α = {alpha})"

    def fit(self, hist: History, t: int) -> ExponentialSmoothing:
        return self

    def predict(self, hist: History, t: int, H: int) -> np.ndarray:
        Y = hist.Y
        level = Y[:, : min(4, t + 1)].mean(axis=1)
        for k in range(t + 1):
            level = level + self.alpha * (Y[:, k] - level)
        return np.repeat(level[:, None], H, axis=1)


class LastYearTrend:
    """Same weeks last year (±1 week), scaled by this year's vs. last year's recent level.

    Falls back to the 8-week mean while there is less than a year of history.
    """

    name = "Vorjahr × Trend"

    def fit(self, hist: History, t: int) -> LastYearTrend:
        return self

    def predict(self, hist: History, t: int, H: int) -> np.ndarray:
        Y = hist.Y
        recent = Y[:, max(0, t - 7) : t + 1].mean(axis=1)
        out = np.repeat(recent[:, None], H, axis=1)
        if t - 64 < 0:
            return out
        trend = Y[:, t - 12 : t + 1].mean(axis=1) / np.maximum(Y[:, t - 64 : t - 51].mean(axis=1), 1)
        trend = np.clip(trend, 0.25, 4.0)
        for h in range(1, H + 1):
            c = t + h - 52
            out[:, h - 1] = Y[:, c - 1 : c + 2].mean(axis=1) * trend
        return out


@dataclass
class GlobalGBM:
    """One LightGBM model across all articles and horizons (horizon is a feature).

    Target is the demand of week t+h divided by the article's recent level; Tweedie loss
    suits non-negative demand with many small and few very large weeks.
    """

    horizon: int = 3
    n_estimators: int = 600
    learning_rate: float = 0.03
    num_leaves: int = 31
    seed: int = 42
    name: str = "LightGBM (global)"
    _model: lgb.LGBMRegressor | None = field(default=None, repr=False)

    def _rows(self, hist: History, t: int):
        X, y = [], []
        for h in range(1, self.horizon + 1):
            for o in range(MIN_HISTORY, t - h + 1):
                X.append(origin_features(hist, o, h))
                y.append(hist.Y[:, o + h] / scale(hist.Y, o))
        return pd.concat(X, ignore_index=True), np.concatenate(y)

    def fit(self, hist: History, t: int) -> GlobalGBM:
        X, y = self._rows(hist, t)
        self._model = lgb.LGBMRegressor(
            objective="tweedie",
            tweedie_variance_power=1.3,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            min_child_samples=100,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            random_state=self.seed,
            verbose=-1,
        ).fit(X, y)
        return self

    def predict(self, hist: History, t: int, H: int) -> np.ndarray:
        s = scale(hist.Y, t)
        cols = [self._model.predict(origin_features(hist, t, h)) * s for h in range(1, H + 1)]
        return np.column_stack(cols)

    def feature_importance(self) -> pd.Series:
        booster = self._model.booster_
        return pd.Series(
            booster.feature_importance("gain"), index=booster.feature_name()
        ).sort_values(ascending=False)
