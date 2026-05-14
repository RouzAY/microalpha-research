# scripts/build_phase4_episode_summary.py
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"

INPUT = REPORTS / "alpha_phase4_results.csv"
OUTPUT = REPORTS / "alpha_phase4_episode_summary.csv"


def mode_or_first(series: pd.Series):
    s = series.dropna()
    if len(s) == 0:
        return None
    m = s.mode()
    if len(m) > 0:
        return m.iloc[0]
    return s.iloc[0]


def build_episode_summary(df: pd.DataFrame) -> pd.DataFrame:
    if "strategy" not in df.columns:
        raise ValueError("alpha_phase4_results.csv must contain a 'strategy' column.")

    # Flexible episode keys depending on what exists in the results file
    candidate_keys = ["asset", "episode", "day", "date", "market", "venue"]
    group_keys = [c for c in candidate_keys if c in df.columns]

    if len(group_keys) == 0:
        raise ValueError(
            "Could not find episode identifiers. Expected at least one of: "
            "asset, episode, day, date, market, venue."
        )

    # Columns that may or may not exist
    num_cols = {
        "net_bps": "net_bps",
        "gross_bps": "gross_bps",
        "cost_bps": "cost_bps",
        "turnover": "turnover",
        "shift_score": "shift_score",
    }
    available_num_cols = {k: v for k, v in num_cols.items() if v in df.columns}

    agg = {}
    for name, col in available_num_cols.items():
        agg[f"mean_{name}"] = (col, "mean")
        agg[f"total_{name}"] = (col, "sum")

    if "mode" in df.columns:
        agg["selected_mode"] = ("mode", mode_or_first)

    if "is_shifted" in df.columns:
        agg["mean_is_shifted"] = ("is_shifted", "mean")

    # number of rows / steps in the episode
    any_col = df.columns[0]
    agg["n_steps"] = (any_col, "size")

    out = (
        df.groupby(group_keys + ["strategy"], dropna=False)
        .agg(**agg)
        .reset_index()
    )

    # Useful derived columns
    if "total_net_bps" in out.columns:
        out["positive_episode"] = (out["total_net_bps"] > 0).astype(int)

    if "mean_is_shifted" in out.columns:
        out["shifted_episode"] = (out["mean_is_shifted"] >= 0.5).astype(int)

    # Reorder columns nicely
    preferred = [
        "asset",
        "market",
        "venue",
        "episode",
        "day",
        "date",
        "strategy",
        "selected_mode",
        "shifted_episode",
        "mean_shift_score",
        "total_net_bps",
        "mean_net_bps",
        "total_gross_bps",
        "mean_gross_bps",
        "total_cost_bps",
        "mean_cost_bps",
        "mean_turnover",
        "total_turnover",
        "n_steps",
        "positive_episode",
    ]
    ordered = [c for c in preferred if c in out.columns]
    remaining = [c for c in out.columns if c not in ordered]
    out = out[ordered + remaining]

    return out.sort_values(
        by=[c for c in ["asset", "episode", "day", "date", "strategy"] if c in out.columns]
    ).reset_index(drop=True)


def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT}")

    df = pd.read_csv(INPUT)
    summary = build_episode_summary(df)
    summary.to_csv(OUTPUT, index=False)

    print(f"Wrote: {OUTPUT}")
    print(summary.head(20).to_string(index=False))


if __name__ == "__main__":
    main()