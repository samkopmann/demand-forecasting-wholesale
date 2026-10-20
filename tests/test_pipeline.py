import numpy as np
import pandas as pd
import pytest

from wholesale_forecast import backtest, data, inventory, metrics, synthetic
from wholesale_forecast.features import origin_features
from wholesale_forecast.models import ExponentialSmoothing, GlobalGBM, MovingAverage


def _tx(rows):
    return pd.DataFrame(rows, columns=["Invoice", "StockCode", "Description", "Quantity",
                                       "InvoiceDate", "Price", "Customer ID", "Country"])


def test_overlapping_sheets_are_not_counted_twice():
    a = _tx([["1", "10001", "A", 1, pd.Timestamp("2010-12-01"), 1.0, 1, "UK"],
             ["2", "10001", "A", 2, pd.Timestamp("2010-12-09"), 1.0, 1, "UK"]])
    b = _tx([["2", "10001", "A", 2, pd.Timestamp("2010-12-09"), 1.0, 1, "UK"],
             ["3", "10001", "A", 4, pd.Timestamp("2010-12-10"), 1.0, 1, "UK"]])
    assert data.combine_sheets([a, b])["Invoice"].tolist() == ["1", "2", "3"]


def test_weekly_panel_nets_cancellations_and_drops_non_products():
    mon = pd.Timestamp("2011-01-03")
    tx = _tx([
        ["1", "10001", "A", 10, mon, 1.0, 1, "UK"],
        ["C2", "10001", "A", -4, mon + pd.Timedelta(days=1), 1.0, 1, "UK"],
        ["3", "10001", "A", 5, mon + pd.Timedelta(days=7), 1.2, 2, "UK"],
        ["4", "POST", "Postage", 1, mon, 18.0, 1, "UK"],
        ["5", "10001", "A", -3, mon + pd.Timedelta(days=8), 0.0, None, "UK"],  # stock correction
        ["6", "10002", "B", 2, mon + pd.Timedelta(days=13), 2.0, 3, "UK"],  # Sunday: week complete
    ])
    panel = data.weekly_panel(tx)
    assert list(panel.demand.index) == ["10001", "10002"]
    assert panel.demand.loc["10001"].tolist() == [6.0, 5.0]
    assert panel.orders.loc["10001"].tolist() == [1.0, 1.0]
    assert panel.price.loc["10001"].tolist() == [1.0, 1.2]


def test_partial_last_week_is_dropped():
    mon = pd.Timestamp("2011-01-03")
    tx = _tx([["1", "10001", "A", 1, mon, 1.0, 1, "UK"],
              ["2", "10001", "A", 1, mon + pd.Timedelta(days=13), 1.0, 1, "UK"],
              ["3", "10001", "A", 1, mon + pd.Timedelta(days=16), 1.0, 1, "UK"]])
    assert len(data.weekly_panel(tx).weeks) == 2


def test_features_do_not_see_the_future():
    hist = synthetic.generate(n_articles=20).history()
    t = 70
    changed = data.History(hist.Y.copy(), hist.orders.copy(), hist.price.copy(), hist.weeks)
    changed.Y[:, t + 1 :] *= 10
    changed.orders[:, t + 1 :] += 5
    changed.price[:, t + 1 :] *= 2
    for h in (1, 3):
        pd.testing.assert_frame_equal(origin_features(hist, t, h), origin_features(changed, t, h))


def test_inventory_simulation_bounds():
    demand = np.array([[5.0, 5.0, 5.0, 5.0]])
    exact = np.full((1, 4, 1), 5.0)
    one = np.ones(1)
    assert inventory.simulate(demand, exact, one, one, lead_time=0, k=0).fill_rate == 1.0
    assert inventory.simulate(demand, exact * 0, one, one, lead_time=0, k=0).fill_rate == 0.0
    low = inventory.simulate(demand, exact * 0.8, one, one, lead_time=0, k=0)
    high = inventory.simulate(demand, exact * 0.8, one, one, lead_time=0, k=3)
    assert low.fill_rate < high.fill_rate and low.avg_stock_value < high.avg_stock_value


def test_orders_arrive_after_lead_time():
    # Only the first decision forecasts demand, for the third week ahead: that order arrives
    # in time for demand in week 2 and is too late for demand in week 1.
    fc = np.zeros((1, 4, 3))
    fc[0, 0, 2] = 4.0
    one = np.ones(1)
    on_time = inventory.simulate(np.array([[0.0, 0.0, 4.0, 0.0]]), fc, one, one, 2, k=0, warmup=0)
    too_late = inventory.simulate(np.array([[0.0, 4.0, 0.0, 0.0]]), fc, one, one, 2, k=0, warmup=0)
    assert on_time.fill_rate == 1.0
    assert too_late.fill_rate == 0.0


def test_exact_forecast_with_lead_time_needs_no_safety_stock():
    demand = np.full((1, 6), 4.0)
    fc = np.full((1, 6, 3), 4.0)
    res = inventory.simulate(demand, fc, np.ones(1), np.ones(1), lead_time=2, k=0, warmup=0)
    assert res.fill_rate == 1.0


def test_safety_factor_search_hits_target():
    rng = np.random.default_rng(1)
    demand = rng.poisson(20, (50, 30)).astype(float)
    fc = np.full((50, 30, 3), 20.0)
    res = inventory.for_fill_rate(0.95, demand, fc, np.full(50, 8.0), np.ones(50), lead_time=2)
    assert res.fill_rate == pytest.approx(0.95, abs=0.005)


def test_metrics():
    y, f = np.array([100.0, 200.0]), np.array([110.0, 180.0])
    assert metrics.wape(y, f) == 30 / 300
    assert metrics.bias(y, f) == -10 / 300


def test_gbm_beats_baselines_on_synthetic_data():
    panel = synthetic.generate(n_articles=60)
    hist = panel.history()
    setup = backtest.Setup(test_weeks=12)
    methods = [MovingAverage(8), ExponentialSmoothing(0.2), GlobalGBM(n_estimators=150)]
    summary, curves = backtest.run(hist, np.ones(60), methods, setup, service_levels=(0.95,),
                                   log=lambda *_: None)
    wape = summary["WAPE 3 Wochen kumuliert"]
    assert wape["LightGBM (global)"] < wape.drop("LightGBM (global)").min()
    assert (summary["Lieferbereitschaft @ 95%"] > 0.94).all()
    assert set(curves["Methode"]) == set(summary.index)
