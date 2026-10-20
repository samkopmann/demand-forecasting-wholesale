"""Forecast error metrics on (articles × periods) arrays, pooled over all articles."""

from __future__ import annotations

import numpy as np


def wape(actual: np.ndarray, forecast: np.ndarray) -> float:
    """Weighted absolute percentage error: sum |error| / sum actual."""
    return float(np.abs(actual - forecast).sum() / np.abs(actual).sum())


def bias(actual: np.ndarray, forecast: np.ndarray) -> float:
    """Relative bias: > 0 means the forecast is systematically too high."""
    return float((forecast - actual).sum() / actual.sum())
