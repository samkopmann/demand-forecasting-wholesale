# Disposition im Großhandel: Wochenprognose und Lagerbestand

Wochenbedarf für **1.016 Artikel eines Großhändlers**, bestellt wird jede Woche bei
zwei Wochen Lieferzeit. Verglichen werden die Verfahren aus Excel und ERP mit einem
globalen LightGBM-Modell. Bewertet wird nicht nur der Prognosefehler, sondern
**wie viel Bestand jedes Verfahren braucht, um dieselbe Lieferbereitschaft zu erreichen**.

> Daten: [Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii)
> (UCI Machine Learning Repository, Daqing Chen, Lizenz CC BY 4.0): alle Rechnungspositionen
> eines britischen Online-Großhändlers für Geschenkartikel, Dez. 2009 – Dez. 2011.
> Viele Kunden sind selbst Händler. `make data` lädt die Daten von UCI.

## Ergebnis

26 wöchentliche Bestellentscheidungen (Nachfragewochen 6. Juni – 4. Dezember 2011, inklusive
Weihnachtsgeschäft), 2 Wochen Lieferzeit:

| Methode | WAPE 3 Wochen | Bias | Ø Bestand bei 95 % Lieferbereitschaft | vs. gleitender Durchschnitt |
|---|---|---|---|---|
| Gleitender Durchschnitt (8 Wochen) | 51,8 % | −6,6 % | £ 866.000 | – |
| Exponentielle Glättung (α = 0,2) | 50,6 % | −6,6 % | £ 782.000 | −9,7 % |
| Vorjahr × Trend | 69,8 % | +17,3 % | £ 1.128.000 | +30,2 % |
| **LightGBM (global)** | **45,8 %** | −9,3 % | **£ 725.000** | **−16,3 %** |

Bestand bei gleicher Lieferbereitschaft:

| Methode | 90 % | 95 % | 98 % |
|---|---|---|---|
| Gleitender Durchschnitt (8 Wochen) | £ 454.000 | £ 866.000 | £ 2.064.000 |
| Exponentielle Glättung (α = 0,2) | £ 446.000 | £ 782.000 | £ 1.510.000 |
| Vorjahr × Trend | £ 722.000 | £ 1.128.000 | £ 2.036.000 |
| **LightGBM (global)** | **£ 394.000** | **£ 725.000** | **£ 1.471.000** |

Einordnung:

- Gegenüber dem gleitenden Durchschnitt braucht das Modell **13–29 % weniger Bestand** für
  dieselbe Lieferbereitschaft, gegenüber der exponentiellen Glättung 3–12 %.
- Der Prognosefehler bleibt hoch (WAPE 46 %). Großhandelsnachfrage ist sprunghaft:
  In einem typischen Artikel ist die stärkste Woche rund zehnmal so hoch wie der Median,
  meist durch einzelne Großaufträge. Diese Sprünge treiben den Sicherheitsbestand.
- „Vorjahr × Trend“ ist hier am schlechtesten: Einzelne Großaufträge im Vorjahr werden
  1:1 in die Zukunft kopiert.
- Bestand ist zu Verkaufspreisen bewertet (Einkaufspreise sind nicht bekannt). Für den
  Vergleich der Verfahren spielt das keine Rolle.

## Fragestellung

Ein Großhändler bestellt wöchentlich bei seinen Lieferanten. Die Bestellmenge ergibt sich aus
der Prognose für die Lieferzeit plus einem Sicherheitsbestand. Frage: **Wie viel Kapital
bindet ein besseres Prognosemodell weniger, ohne dass die Lieferfähigkeit leidet?**

## Vorgehen

| Schritt | Umsetzung |
|---|---|
| Daten | Rechnungspositionen → Nettomenge je Artikel und Woche. Stornos werden mit Bestellungen derselben Woche verrechnet. Porto, Gebühren und Korrekturbuchungen werden entfernt, ebenso die doppelt enthaltenen Tage 1.–9.12.2010 |
| Artikelauswahl | Artikel mit Nachfrage in mindestens 70 % der Wochen **vor** dem Test (Schnell- und Mitteldreher). Neue Artikel und Sporadiker brauchen eigene Verfahren (z. B. Croston) |
| Baselines | ① gleitender Durchschnitt 8 Wochen (Excel) ② exponentielle Glättung (ERP-Standard) ③ gleiche Wochen im Vorjahr × aktueller Trend |
| Modell | Ein LightGBM über alle Artikel und Horizonte (1–3 Wochen), Tweedie-Verlust. Features: Lags, Mittelwerte, Anzahl Bestellpositionen, Vorjahresverlauf, Preis, Kalenderwoche. Alles relativ zum Artikelniveau. Nachtraining alle 4 Wochen |
| Backtest | Rollierend: Jede Prognose sieht nur Daten bis zur jeweiligen Bestellentscheidung. Ein Test prüft das, indem er die Zukunft verändert und gleiche Features erwartet |
| Lagersimulation | Bestellpunkt-Politik mit wöchentlicher Überprüfung: Bestellung auf Prognose für Lieferzeit + 1 Woche plus k · σ. σ ist der Prognosefehler **des jeweiligen Verfahrens** je Artikel in den 13 Wochen vor dem Test. k wird je Verfahren so gewählt, dass alle dieselbe Lieferbereitschaft erreichen. Nicht lieferbare Nachfrage geht verloren |
| Metriken | WAPE der Folgewoche und der kumulierten 3 Wochen, Bias, Ø Lagerwert bei 90 / 95 / 98 % Lieferbereitschaft (mengengewichtet) |

Methodik und Modell wurden auf einem früheren Zeitraum entwickelt (Test bis Ende Mai 2011).
Der hier gezeigte Testzeitraum wurde nicht zum Feinabstimmen genutzt.

## Reproduzieren

```bash
# Voraussetzungen: Python ≥ 3.11, uv (https://docs.astral.sh/uv/)
uv sync --extra dev
make demo        # läuft sofort mit synthetischen Daten (ohne Download)
make data        # Online Retail II von UCI laden (~45 MB)
make backtest    # ca. 2 Minuten; schneller: uv run wholesale-forecast backtest --articles 200
make test
```

`results/backtest.md` enthält die Tabellen, `results/tradeoff.csv` die Kurve Bestand gegen
Lieferbereitschaft je Verfahren.

## Aufbau

```
src/wholesale_forecast/
  data.py       Download, Bereinigung, Wochenaggregation, Artikelauswahl
  features.py   Features zum Prognosezeitpunkt (ohne Zukunftswissen)
  models.py     Baselines + globales LightGBM
  backtest.py   rollierender Backtest, Kalibrierung, Auswertung
  inventory.py  Lagersimulation mit Lieferzeit, Sicherheitsfaktor je Lieferbereitschaft
  synthetic.py  synthetische Daten im selben Format (Tests, Demo)
```

## Übertragbarkeit

Jeder Großhändler hat diese Daten im ERP: Rechnungs- oder Lieferscheinpositionen je Artikel
und Datum. Das Vorgehen lässt sich direkt übertragen. Der Vergleich findet auf Ihren eigenen
Daten statt, gegen Ihr aktuelles Verfahren, bevor irgendetwas produktiv geht.

---
Dr. Samuel Kopmann · [softwareandmore.de/forecasting](https://softwareandmore.de/forecasting)

© 2026 Dr. Samuel Kopmann. Alle Rechte vorbehalten. Der Code ist zum Ansehen und Nachrechnen der
Ergebnisse veröffentlicht; für eine Nutzung in eigenen Projekten sprechen Sie mich gern an.
