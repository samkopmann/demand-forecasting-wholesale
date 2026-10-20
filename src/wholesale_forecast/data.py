"""Online Retail II (UCI): all transactions of a UK online wholesaler, Dec 2009 – Dec 2011.

Source: https://archive.ics.uci.edu/dataset/502/online+retail+ii (Daqing Chen, 2012),
licence CC BY 4.0. The Excel file has two sheets that overlap by nine days
(2010-12-01 to 2010-12-09); the overlap is taken from the first sheet only.
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests

URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
RAW_DIR = Path("data/raw")
CACHE = Path("data/transactions.pkl")

# Postage, fees, manual adjustments, gift vouchers etc. are not products.
PRODUCT_CODE = r"^\d{5}"


def download(raw_dir: Path = RAW_DIR) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    resp = requests.get(URL, timeout=300)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        zf.extractall(raw_dir)


def load_transactions(raw_dir: Path = RAW_DIR, cache: Path = CACHE) -> pd.DataFrame:
    """All invoice lines; reading the Excel file takes a minute, so the result is cached."""
    if cache.exists():
        return pd.read_pickle(cache)
    sheets = pd.read_excel(
        raw_dir / "online_retail_II.xlsx",
        sheet_name=None,
        dtype={"Invoice": str, "StockCode": str},
    )
    tx = combine_sheets(list(sheets.values()))
    tx.to_pickle(cache)
    return tx


def combine_sheets(sheets: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate yearly sheets, dropping rows already covered by the previous sheet."""
    parts, covered_until = [], pd.Timestamp.min
    for sheet in sheets:
        parts.append(sheet[sheet["InvoiceDate"] > covered_until])
        covered_until = sheet["InvoiceDate"].max()
    return pd.concat(parts, ignore_index=True)


@dataclass
class Panel:
    """Weekly data per article: `demand`, `orders` (order lines) and `price` are
    (articles × weeks)."""

    demand: pd.DataFrame
    orders: pd.DataFrame
    price: pd.DataFrame
    description: pd.Series

    @property
    def weeks(self) -> pd.DatetimeIndex:
        return self.demand.columns

    def subset(self, articles) -> Panel:
        return Panel(self.demand.loc[articles], self.orders.loc[articles],
                     self.price.loc[articles], self.description.loc[articles])

    def history(self) -> History:
        return History(self.demand.to_numpy(dtype=float), self.orders.to_numpy(dtype=float),
                       self.price.to_numpy(dtype=float), self.weeks)


@dataclass
class History:
    """Arrays the forecasting methods see. `upto(t)` cuts everything after week t, so a
    method cannot use information that was not available at the forecast origin."""

    Y: np.ndarray
    orders: np.ndarray
    price: np.ndarray
    weeks: pd.DatetimeIndex  # kept complete: target dates lie beyond the origin

    def upto(self, t: int) -> History:
        return History(self.Y[:, : t + 1], self.orders[:, : t + 1], self.price[:, : t + 1],
                       self.weeks)


def weekly_panel(tx: pd.DataFrame, drop_partial_last_week: bool = True) -> Panel:
    """Net weekly quantity per article.

    Cancellations (invoice prefix "C") carry negative quantities and are netted against the
    orders of the same week, so orders that were entered and cancelled right away do not count
    as demand. Lines with price <= 0 are stock corrections, not sales.
    """
    tx = tx[tx["StockCode"].astype(str).str.match(PRODUCT_CODE) & (tx["Price"] > 0)].copy()
    tx["StockCode"] = tx["StockCode"].astype(str).str.upper()
    tx["week"] = tx["InvoiceDate"].dt.to_period("W-SUN").dt.start_time

    weeks = pd.date_range(tx["week"].min(), tx["week"].max(), freq="W-MON")
    if drop_partial_last_week and tx["InvoiceDate"].max() < weeks[-1] + pd.Timedelta(days=6):
        weeks = weeks[:-1]
        tx = tx[tx["week"].isin(weeks)]

    demand = (
        tx.groupby(["StockCode", "week"])["Quantity"].sum().clip(lower=0)
        .unstack(fill_value=0).reindex(columns=weeks, fill_value=0)
    )
    sales = tx[tx["Quantity"] > 0]
    orders = (
        sales.groupby(["StockCode", "week"]).size()
        .unstack(fill_value=0).reindex(index=demand.index, columns=weeks, fill_value=0)
    )
    price = (
        sales.groupby(["StockCode", "week"])["Price"].median()
        .unstack().reindex(index=demand.index, columns=weeks).ffill(axis=1).bfill(axis=1)
    )
    description = sales.groupby("StockCode")["Description"].agg(
        lambda s: s.mode().iat[0] if s.notna().any() else ""
    )
    return Panel(demand.astype(float), orders.astype(float), price,
                 description.reindex(demand.index))


def select_articles(panel: Panel, until: int, min_active: float = 0.7) -> Panel:
    """Regular sellers: articles with demand in at least `min_active` of the weeks before
    index `until`. Only past weeks are used, so the selection does not peek into the test."""
    active = (panel.demand.iloc[:, :until] > 0).mean(axis=1) >= min_active
    return panel.subset(active[active].index)
