# Backtest-Ergebnis

- Daten: Online Retail II (UCI, CC BY 4.0), 1016 Artikel, Wochen 2009-11-30 bis 2011-11-28
- Test: 26 Bestellentscheidungen, Nachfragewochen ab 2011-06-06; Lieferzeit 2 Wochen, wöchentliche Bestellung
- Sicherheitsbestand aus den Prognosefehlern der 13 Wochen davor; Bestand bewertet zu Verkaufspreisen

## Prognosegüte und Bestand bei 95 % Lieferbereitschaft

| Methode | WAPE Folgewoche | WAPE 3 Wochen | Bias | Ø Bestand @ 95 % | vs. Gleitender Durchschnitt |
|---|---|---|---|---|---|
| Gleitender Durchschnitt (8 Wochen) | 66.3% | 51.8% | -6.6% | £ 866,176 | +0.0% |
| Exponentielle Glättung (α = 0.2) | 65.2% | 50.6% | -6.6% | £ 782,269 | -9.7% |
| Vorjahr × Trend | 83.5% | 69.8% | +17.3% | £ 1,127,607 | +30.2% |
| LightGBM (global) | 62.2% | 45.8% | -9.3% | £ 725,376 | -16.3% |

## Bestand bei gleicher Lieferbereitschaft

| Methode | Bestand @ 90% | Bestand @ 95% | Bestand @ 98% |
|---|---|---|---|
| Gleitender Durchschnitt (8 Wochen) | £ 453,635 | £ 866,176 | £ 2,064,394 |
| Exponentielle Glättung (α = 0.2) | £ 445,863 | £ 782,269 | £ 1,510,135 |
| Vorjahr × Trend | £ 722,045 | £ 1,127,607 | £ 2,036,362 |
| LightGBM (global) | £ 393,869 | £ 725,376 | £ 1,471,456 |

Wichtigste Merkmale (LightGBM): active13, mean52, target_weekofyear, log_level, mean26, log_price, weeks_since_sale, ly_growth_to_target
