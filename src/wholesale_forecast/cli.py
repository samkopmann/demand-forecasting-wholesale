"""Command line: `wholesale-forecast download | backtest [--demo] [--articles N]`."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from . import backtest, data, synthetic
from .models import ExponentialSmoothing, GlobalGBM, LastYearTrend, MovingAverage

RESULTS = Path("results")


def _load(args, setup: backtest.Setup) -> data.Panel:
    if args.demo:
        return synthetic.generate(n_articles=args.articles or 200)
    panel = data.weekly_panel(data.load_transactions())
    calib, _ = setup.origins(panel.demand.shape[1])
    panel = data.select_articles(panel, until=calib[0] + 1, min_active=args.min_active)
    if args.articles:
        keep = np.random.default_rng(0).choice(len(panel.demand), args.articles, replace=False)
        panel = panel.subset(panel.demand.index[keep])
    return panel


def cmd_backtest(args) -> None:
    setup = backtest.Setup(lead_time=args.lead_time, test_weeks=args.test_weeks)
    panel = _load(args, setup)
    _, test = setup.origins(panel.demand.shape[1])
    hist, weeks = panel.history(), panel.weeks
    # Stock is valued at the median selling price before the test (purchase prices are unknown).
    unit_value = np.median(hist.price[:, : test[0] + 1], axis=1)
    print(f"{len(panel.demand)} Artikel, Wochen {weeks[0].date()} bis {weeks[-1].date()}, "
          f"Test ab {weeks[test[0] + 1].date()}, Lieferzeit {setup.lead_time} Wochen")

    methods = [
        MovingAverage(8),
        ExponentialSmoothing(0.2),
        LastYearTrend(),
        GlobalGBM(horizon=setup.horizon, n_estimators=args.trees),
    ]
    summary, curves = backtest.run(hist, unit_value, methods, setup)

    RESULTS.mkdir(exist_ok=True)
    suffix = "_demo" if args.demo else ""
    summary.to_csv(RESULTS / f"backtest{suffix}.csv")
    curves.to_csv(RESULTS / f"tradeoff{suffix}.csv", index=False)
    importance = methods[-1].feature_importance()
    (RESULTS / f"backtest{suffix}.md").write_text(
        _markdown(summary, importance, panel, setup, args), encoding="utf-8"
    )
    print("\n" + summary.to_string(float_format=lambda v: f"{v:,.3f}"))
    print(f"\n→ {RESULTS / f'backtest{suffix}.md'}")


def _markdown(summary, importance, panel, setup, args) -> str:
    H = setup.horizon
    base = summary.index[0]
    lines = [
        (f"| Methode | WAPE Folgewoche | WAPE {H} Wochen | Bias | Ø Bestand @ 95 % "
         f"| vs. {base.split(' (')[0]} |"),
        "|---|---|---|---|---|---|",
    ]
    for name, r in summary.iterrows():
        rel = r["Bestand @ 95%"] / summary.loc[base, "Bestand @ 95%"] - 1
        lines.append(
            f"| {name} | {r['WAPE Folgewoche']:.1%} | {r[f'WAPE {H} Wochen kumuliert']:.1%} | "
            f"{r['Bias']:+.1%} | £ {r['Bestand @ 95%']:,.0f} | {rel:+.1%} |"
        )
    levels = [c.split("@ ")[1] for c in summary.columns if c.startswith("Bestand @")]
    sl = ["| Methode | " + " | ".join(f"Bestand @ {lv}" for lv in levels) + " |",
          "|---|" + "---|" * len(levels)]
    for name, r in summary.iterrows():
        sl.append(f"| {name} | " + " | ".join(f"£ {r[f'Bestand @ {lv}']:,.0f}" for lv in levels)
                  + " |")
    src = "synthetische Demodaten" if args.demo else "Online Retail II (UCI, CC BY 4.0)"
    weeks = panel.weeks
    _, test = setup.origins(len(weeks))
    top = ", ".join(importance.head(8).index)
    return (
        "# Backtest-Ergebnis\n\n"
        f"- Daten: {src}, {len(panel.demand)} Artikel, Wochen {weeks[0].date()} bis "
        f"{weeks[-1].date()}\n"
        f"- Test: {setup.test_weeks} Bestellentscheidungen, Nachfragewochen ab "
        f"{weeks[test[0] + 1].date()}; Lieferzeit {setup.lead_time} Wochen, "
        f"wöchentliche Bestellung\n"
        f"- Sicherheitsbestand aus den Prognosefehlern der {setup.calib_weeks} Wochen davor; "
        "Bestand bewertet zu Verkaufspreisen\n\n"
        "## Prognosegüte und Bestand bei 95 % Lieferbereitschaft\n\n" + "\n".join(lines) +
        "\n\n## Bestand bei gleicher Lieferbereitschaft\n\n" + "\n".join(sl) +
        f"\n\nWichtigste Merkmale (LightGBM): {top}\n"
    )


def main() -> None:
    p = argparse.ArgumentParser(prog="wholesale-forecast")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("download", help="Online Retail II von UCI laden").set_defaults(
        func=lambda _: data.download()
    )
    b = sub.add_parser("backtest", help="Backtest und Lagersimulation aller Methoden")
    b.add_argument("--demo", action="store_true", help="synthetische Daten statt UCI")
    b.add_argument("--articles", type=int, help="nur N zufällige Artikel (schneller)")
    b.add_argument("--min-active", type=float, default=0.7,
                   help="Mindestanteil Wochen mit Nachfrage vor dem Test")
    b.add_argument("--lead-time", type=int, default=2, help="Lieferzeit in Wochen")
    b.add_argument("--test-weeks", type=int, default=26)
    b.add_argument("--trees", type=int, default=600)
    b.set_defaults(func=cmd_backtest)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
