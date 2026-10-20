"""What a forecast is worth in the warehouse: weekly order-up-to simulation with lead time.

Every week, after the week's sales are known, the planner orders up to
    S = forecast for the next L+1 weeks + k · σ
where L is the supplier lead time in weeks and σ is the standard deviation of the method's own
(L+1)-week forecast errors, measured per article in a calibration period before the test.
Orders arrive L weeks later; demand that cannot be served from stock is lost.

Better forecasts need less safety stock for the same service. To compare methods fairly, the
safety factor k is set per method so that all methods reach the same fill rate, and then the
average stock value is compared.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Result:
    k: float
    fill_rate: float
    avg_stock_value: float


def simulate(demand: np.ndarray, forecast: np.ndarray, sigma: np.ndarray, unit_value: np.ndarray,
             lead_time: int, k: float, warmup: int | None = None) -> Result:
    """
    demand:   (articles × weeks) demand of the weeks after each order decision
    forecast: (articles × weeks × (L+1)) forecasts made at each decision for the next L+1 weeks
    sigma:    (articles,) std. dev. of the (L+1)-week forecast error
    """
    n, m, H = forecast.shape
    assert H == lead_time + 1 and demand.shape == (n, m)
    warmup = lead_time if warmup is None else warmup
    safety = k * sigma
    # Steady-state start: safety stock on hand, the next L weeks' forecast already in transit.
    on_hand = safety.copy()
    pipeline = [forecast[:, 0, h].copy() for h in range(lead_time)]
    filled = demand_total = stock_value = 0.0
    for j in range(m):
        position = on_hand + sum(pipeline, np.zeros(n))
        target = forecast[:, j, :].sum(axis=1) + safety
        pipeline.append(np.maximum(target - position, 0.0))
        on_hand = on_hand + pipeline.pop(0)
        sold = np.minimum(on_hand, demand[:, j])
        if j >= warmup:
            filled += sold.sum()
            demand_total += demand[:, j].sum()
            stock_value += ((on_hand - sold / 2) * unit_value).sum()
        on_hand = on_hand - sold
    weeks = m - warmup
    return Result(k, filled / demand_total if demand_total else float("nan"), stock_value / weeks)


def for_fill_rate(target: float, demand, forecast, sigma, unit_value, lead_time,
                  k_max: float = 20.0, tol: float = 1e-4) -> Result:
    """Smallest safety factor k reaching the target fill rate (bisection)."""
    def run(k: float) -> Result:
        return simulate(demand, forecast, sigma, unit_value, lead_time, k)

    hi = run(k_max)
    if hi.fill_rate < target:
        return hi
    lo_k, hi_k = 0.0, k_max
    if run(0.0).fill_rate >= target:
        return run(0.0)
    while hi_k - lo_k > tol:
        mid = (lo_k + hi_k) / 2
        if run(mid).fill_rate >= target:
            hi_k = mid
        else:
            lo_k = mid
    return run(hi_k)


def error_sigma(actual_cum: np.ndarray, forecast_cum: np.ndarray, floor: float = 1.0) -> np.ndarray:
    """Per-article RMSE of cumulative (L+1)-week forecasts; inputs are (articles × origins)."""
    return np.maximum(np.sqrt(np.mean((actual_cum - forecast_cum) ** 2, axis=1)), floor)
