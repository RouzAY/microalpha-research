from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import glob
from typing import Iterable

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "spread",
    "imb1",
    "imb2",
    "micro_dev",
    "trade_signed_roll5",
    "trade_signed_roll15",
    "trade_abs_roll5",
    "trade_abs_roll15",
    "ntrades_roll5",
    "ret_5",
    "ret_15",
    "ret_30",
]


@dataclass
class DayData:
    day: str
    symbol: str
    book: pd.DataFrame


class DataError(RuntimeError):
    pass


def list_available_days(data_root: str | Path, symbol: str = "AAPL") -> list[str]:
    data_root = Path(data_root)
    matches = sorted((data_root / "quotes_depth").glob(f"date=*/symbol={symbol}"))
    days = sorted(path.parent.name.split("=", 1)[1] for path in matches if path.is_dir())
    if not days:
        raise DataError(f"No days found under {data_root}")
    return days


def _single_file(pattern: str) -> str:
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise DataError(f"No file matched pattern: {pattern}")
    return matches[0]


def load_day(data_root: str | Path, day: str, symbol: str = "AAPL") -> DayData:
    data_root = Path(data_root)
    depth_path = _single_file(str(data_root / "quotes_depth" / f"date={day}" / f"symbol={symbol}" / "*.csv"))
    trades_path = _single_file(str(data_root / "trades" / f"date={day}" / f"symbol={symbol}" / "*.csv"))

    depth = pd.read_csv(depth_path)
    trades = pd.read_csv(trades_path)

    depth["side_name"] = depth["side"].map({1: "bid", -1: "ask"})
    book = depth.pivot_table(
        index="ts_ns",
        columns=["side_name", "level"],
        values=["price", "size"],
        aggfunc="first",
    )
    book.columns = ["_".join([a, b, str(c)]) for a, b, c in book.columns]
    book = book.sort_index()

    book["mid"] = (book["price_bid_1"] + book["price_ask_1"]) / 2.0
    book["spread"] = book["price_ask_1"] - book["price_bid_1"]
    book["imb1"] = (book["size_bid_1"] - book["size_ask_1"]) / (
        book["size_bid_1"] + book["size_ask_1"] + 1e-9
    )
    book["imb2"] = (book["size_bid_2"] - book["size_ask_2"]) / (
        book["size_bid_2"] + book["size_ask_2"] + 1e-9
    )
    book["micro"] = (
        book["price_ask_1"] * book["size_bid_1"] + book["price_bid_1"] * book["size_ask_1"]
    ) / (book["size_bid_1"] + book["size_ask_1"] + 1e-9)
    book["micro_dev"] = book["micro"] - book["mid"]

    trades = trades.copy()
    trades["signed_size"] = np.where(trades["aggressor_side"].eq("BUY"), trades["size"], -trades["size"])
    agg = trades.groupby("ts_ns").agg(
        trade_signed=("signed_size", "sum"),
        trade_abs=("size", "sum"),
        ntrades=("size", "size"),
    )
    book = book.join(agg, how="left").fillna({"trade_signed": 0.0, "trade_abs": 0.0, "ntrades": 0})

    for col in ["trade_signed", "trade_abs", "ntrades"]:
        for window in [5, 15, 30]:
            book[f"{col}_roll{window}"] = book[col].rolling(window, min_periods=1).sum()

    for window in [5, 15, 30]:
        book[f"ret_{window}"] = book["mid"].pct_change(window).fillna(0.0)

    book = book.reset_index(drop=False)
    return DayData(day=day, symbol=symbol, book=book)


def load_many_days(data_root: str | Path, days: Iterable[str], symbol: str = "AAPL") -> dict[str, DayData]:
    return {day: load_day(data_root, day=day, symbol=symbol) for day in days}
