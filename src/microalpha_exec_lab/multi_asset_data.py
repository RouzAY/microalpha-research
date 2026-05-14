from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import glob

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


@dataclass
class EpisodeData:
    asset: str
    venue: str
    day: str
    book: pd.DataFrame


class DataError(RuntimeError):
    pass


AAPL_FEATURE_COLUMNS = [
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


def _single_file(pattern: str) -> str:
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise DataError(f"No file matched pattern: {pattern}")
    return matches[0]


def list_available_days_optiver(data_root: str | Path, symbol: str = "AAPL") -> list[str]:
    data_root = Path(data_root)
    matches = sorted((data_root / "quotes_depth").glob(f"date=*/symbol={symbol}"))
    days = sorted(path.parent.name.split("=", 1)[1] for path in matches if path.is_dir())
    if not days:
        raise DataError(f"No Optiver days found under {data_root}")
    return days


def list_available_days_binance(data_root: str | Path, symbol: str = "BNBUSDT") -> list[str]:
    data_root = Path(data_root)
    matches = sorted((data_root / "quotes_depth").glob(f"date=*/symbol={symbol}"))
    days = sorted(path.parent.name.split("=", 1)[1] for path in matches if path.is_dir())
    if not days:
        raise DataError(f"No Binance days found under {data_root}")
    return days


def _read_parquet_selected(path: str | Path, columns: list[str]) -> pd.DataFrame:
    pf = pq.ParquetFile(path)
    chunks: list[pd.DataFrame] = []
    for idx in range(pf.num_row_groups):
        chunks.append(pf.read_row_group(idx, columns=columns).to_pandas())
    return pd.concat(chunks, ignore_index=True)


def _tick_rule_sign(prices: np.ndarray) -> np.ndarray:
    prices = np.asarray(prices, dtype=float)
    diff = np.diff(prices, prepend=prices[0])
    sign = np.sign(diff)
    last = 1.0
    out = np.empty_like(sign)
    for i, value in enumerate(sign):
        if value == 0.0:
            out[i] = last
        else:
            out[i] = value
            last = value
    return out


def load_optiver_day(data_root: str | Path, day: str, symbol: str = "AAPL") -> EpisodeData:
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

    for col in ["price_bid_2", "price_ask_2", "size_bid_2", "size_ask_2"]:
        if col not in book.columns:
            fallback = col[:-1] + "1"
            book[col] = book[fallback]

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
    return EpisodeData(asset=symbol, venue="XNAS", day=day, book=book)


def load_binance_day(
    data_root: str | Path,
    day: str,
    symbol: str = "BNBUSDT",
    sample_seconds: int = 1,
) -> EpisodeData:
    data_root = Path(data_root)
    depth_path = _single_file(str(data_root / "quotes_depth" / f"date={day}" / f"symbol={symbol}" / "*.parquet"))
    trades_path = _single_file(str(data_root / "trades" / f"date={day}" / f"symbol={symbol}" / "*.parquet"))

    depth = _read_parquet_selected(depth_path, ["ts_ns", "level", "side", "price", "size"])
    depth = depth[depth["level"] == 1].copy()
    depth["sec"] = pd.to_datetime(depth["ts_ns"]).dt.floor(f"{sample_seconds}s")
    quotes = depth.pivot_table(index="sec", columns="side", values=["price", "size"], aggfunc="last")
    quotes.columns = [f"{name}_{'bid' if side == 1 else 'ask'}_1" for name, side in quotes.columns]
    quotes = quotes.sort_index()
    quotes = quotes.dropna(subset=["price_bid_1", "price_ask_1"])

    quotes["mid"] = (quotes["price_bid_1"] + quotes["price_ask_1"]) / 2.0
    quotes["spread"] = quotes["price_ask_1"] - quotes["price_bid_1"]
    quotes["imb1"] = (quotes["size_bid_1"] - quotes["size_ask_1"]) / (
        quotes["size_bid_1"] + quotes["size_ask_1"] + 1e-9
    )
    quotes["imb2"] = quotes["imb1"]
    quotes["micro"] = (
        quotes["price_ask_1"] * quotes["size_bid_1"] + quotes["price_bid_1"] * quotes["size_ask_1"]
    ) / (quotes["size_bid_1"] + quotes["size_ask_1"] + 1e-9)
    quotes["micro_dev"] = quotes["micro"] - quotes["mid"]

    trades = _read_parquet_selected(trades_path, ["ts_ns", "price", "size"]).sort_values("ts_ns").copy()
    trades["sec"] = pd.to_datetime(trades["ts_ns"]).dt.floor(f"{sample_seconds}s")
    trades["signed_size"] = _tick_rule_sign(trades["price"].to_numpy(dtype=float)) * trades["size"].to_numpy(dtype=float)
    agg = trades.groupby("sec").agg(
        trade_signed=("signed_size", "sum"),
        trade_abs=("size", "sum"),
        ntrades=("size", "size"),
    )

    book = quotes.join(agg, how="left").fillna({"trade_signed": 0.0, "trade_abs": 0.0, "ntrades": 0})
    for col in ["trade_signed", "trade_abs", "ntrades"]:
        for window in [5, 15, 30]:
            book[f"{col}_roll{window}"] = book[col].rolling(window, min_periods=1).sum()

    for window in [5, 15, 30]:
        book[f"ret_{window}"] = book["mid"].pct_change(window).fillna(0.0)

    book = book.reset_index(drop=False)
    book["ts_ns"] = book["sec"].astype("int64")
    return EpisodeData(asset=symbol, venue="BINANCE", day=day, book=book)
