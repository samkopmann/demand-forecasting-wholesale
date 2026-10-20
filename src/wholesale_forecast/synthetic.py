"""Synthetic weekly wholesale demand — for tests and a quick demo without the download.

Gift-ware pattern: a strong run-up to Christmas, a closed week around New Year, per-article
trends, Poisson-gamma noise and occasional large one-off orders.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import Panel


def generate(n_articles: int = 200, n_weeks: int = 104, seed: int = 0) -> Panel:
    rng = np.random.default_rng(seed)
    weeks = pd.date_range("2009-11-30", periods=n_weeks, freq="W-MON")
    woy = weeks.isocalendar().week.to_numpy()
    christmas = 1 + 1.2 * np.clip((woy - 34) / 14, 0, 1) * (woy <= 50)
    closed = np.isin(woy, [52, 53, 1])
    season = np.where(closed, 0.1, christmas)

    level = rng.lognormal(4, 1, n_articles)[:, None]
    seasonality = rng.uniform(0.3, 1.0, n_articles)[:, None]
    trend = np.exp(rng.normal(0, 0.006, n_articles)[:, None] * np.arange(n_weeks))
    mean = level * (1 + seasonality * (season - 1)) * trend
    demand = rng.poisson(rng.gamma(4, mean / 4))
    big = rng.random(demand.shape) < 0.03
    demand = demand + big * rng.poisson(level * rng.uniform(3, 15, demand.shape))
    orders = rng.binomial(demand, 1 / rng.uniform(5, 30, n_articles)[:, None]) + big

    codes = [f"{10000 + i}" for i in range(n_articles)]
    base_price = rng.lognormal(0.5, 0.7, n_articles)[:, None]
    price = base_price * np.ones((1, n_weeks))
    return Panel(
        pd.DataFrame(demand.astype(float), index=codes, columns=weeks),
        pd.DataFrame(np.minimum(orders, demand).astype(float), index=codes, columns=weeks),
        pd.DataFrame(price, index=codes, columns=weeks),
        pd.Series([f"ARTIKEL {c}" for c in codes], index=codes),
    )
