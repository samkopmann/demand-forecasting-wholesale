"""Rolling weekly backtest: at every decision week, forecast the next L+1 weeks with data up to
that week only, then score the forecasts and replay the orders in the inventory simulation.

Timeline (t = index of the last complete week at a decision):
    calibration origins   errors are known before the test starts → safety stock σ per method
    test origins          scored; their demand weeks are the last `test_weeks` weeks of the data
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import inventory, metrics
from .data import History


@dataclass
class Setup:
    lead_time: int = 2
    test_weeks: int = 26
    calib_weeks: int = 13
    refit_every: int = 4

    @property
    def horizon(self) -> int:
        return self.lead_time + 1

    def origins(self, n_weeks: int) -> tuple[np.ndarray, np.ndarray]:
        test = np.arange(n_weeks - 1 - self.test_weeks, n_weeks - 1)
        last_calib = test[0] - self.horizon  # its (L+1)-week actuals are known at test start
        calib = np.arange(last_calib - self.calib_weeks + 1, last_calib + 1)
        return calib, test


def rolling_forecasts(hist: History, method, origins: np.ndarray, H: int, refit_every: int,
                      log=print) -> np.ndarray:
    """Forecasts (articles × origins × H); each uses only weeks up to its origin."""
    out = np.empty((hist.Y.shape[0], len(origins), H))
    for j, t in enumerate(origins):
        past = hist.upto(t)
        if j % refit_every == 0:
            method.fit(past, t)
            log(f"  {method.name}: Training bis {hist.weeks[t].date()}")
        out[:, j, :] = method.predict(past, t, H)
    return np.maximum(out, 0.0)


def cumulative_actuals(Y: np.ndarray, origins: np.ndarray, H: int) -> np.ndarray:
    """(articles × origins) demand of weeks t+1..t+H; NaN where it lies beyond the data."""
    out = np.full((Y.shape[0], len(origins)), np.nan)
    for j, t in enumerate(origins):
        if t + H < Y.shape[1]:
            out[:, j] = Y[:, t + 1 : t + H + 1].sum(axis=1)
    return out


def run(hist: History, unit_value: np.ndarray, methods: list, setup: Setup,
        service_levels=(0.90, 0.95, 0.98), log=print) -> tuple[pd.DataFrame, pd.DataFrame]:
    Y = hist.Y
    calib, test = setup.origins(Y.shape[1])
    H = setup.horizon
    origins = np.arange(calib[0], test[-1] + 1)
    i_calib = np.searchsorted(origins, calib)
    i_test = np.searchsorted(origins, test)
    actual_cum = cumulative_actuals(Y, origins, H)
    next_week = Y[:, test + 1]
    scored = ~np.isnan(actual_cum[0, i_test])

    summary, curves = [], []
    for method in methods:
        F = rolling_forecasts(hist, method, origins, H, setup.refit_every, log)
        sigma = inventory.error_sigma(actual_cum[:, i_calib], F[:, i_calib, :].sum(axis=2))
        F_test = F[:, i_test, :]
        cum_a = actual_cum[:, i_test][:, scored]
        cum_f = F_test.sum(axis=2)[:, scored]
        row = {
            "Methode": method.name,
            "WAPE Folgewoche": metrics.wape(next_week, F_test[:, :, 0]),
            f"WAPE {H} Wochen kumuliert": metrics.wape(cum_a, cum_f),
            "Bias": metrics.bias(cum_a, cum_f),
        }
        for sl in service_levels:
            res = inventory.for_fill_rate(sl, next_week, F_test, sigma, unit_value, setup.lead_time)
            row[f"Bestand @ {sl:.0%}"] = res.avg_stock_value
            row[f"Lieferbereitschaft @ {sl:.0%}"] = res.fill_rate
            row[f"k @ {sl:.0%}"] = res.k
        log(f"  → WAPE {row[f'WAPE {H} Wochen kumuliert']:.1%}, "
            f"Bestand @ 95 %: {row.get('Bestand @ 95%', float('nan')):,.0f}")
        summary.append(row)
        for k in np.linspace(0, 6, 25):
            r = inventory.simulate(next_week, F_test, sigma, unit_value, setup.lead_time, k)
            curves.append({"Methode": method.name, "k": k, "Lieferbereitschaft": r.fill_rate,
                           "Bestand": r.avg_stock_value})
    return pd.DataFrame(summary).set_index("Methode"), pd.DataFrame(curves)
