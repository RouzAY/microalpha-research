# app/common.py

from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
DOCS = ROOT / "docs"


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 2rem;
            max-width: 1280px;
        }

        .hero {
            padding: 1.2rem 1.25rem 1rem 1.25rem;
            border: 1px solid rgba(49, 51, 63, 0.12);
            border-radius: 18px;
            background: linear-gradient(180deg, rgba(248,249,252,1) 0%, rgba(255,255,255,1) 100%);
            margin-bottom: 1rem;
        }

        .hero h1 {
            margin: 0 0 0.25rem 0;
            font-size: 2rem;
            font-weight: 700;
        }

        .hero p {
            margin: 0.15rem 0 0 0;
            color: #4b5563;
            line-height: 1.5;
        }

        .section-title {
            margin-top: 1.2rem;
            margin-bottom: 0.35rem;
            font-size: 1.2rem;
            font-weight: 700;
        }

        .muted {
            color: #6b7280;
        }

        .phase-card {
            border: 1px solid rgba(49, 51, 63, 0.12);
            border-radius: 16px;
            padding: 1rem 1rem 0.9rem 1rem;
            background: #ffffff;
            min-height: 210px;
            box-shadow: 0 2px 10px rgba(15, 23, 42, 0.04);
        }

        .phase-card h3 {
            margin-top: 0.25rem;
            margin-bottom: 0.55rem;
            font-size: 1.05rem;
        }

        .phase-card p {
            margin: 0.3rem 0;
            line-height: 1.45;
            color: #374151;
        }

        .badge {
            display: inline-block;
            padding: 0.2rem 0.55rem;
            border-radius: 999px;
            font-size: 0.76rem;
            font-weight: 600;
            margin-right: 0.35rem;
            margin-bottom: 0.25rem;
        }

        .badge-real {
            background: rgba(16, 185, 129, 0.12);
            color: #047857;
        }

        .badge-synth {
            background: rgba(59, 130, 246, 0.12);
            color: #1d4ed8;
        }

        .badge-external {
            background: rgba(168, 85, 247, 0.12);
            color: #7c3aed;
        }

        .badge-method {
            background: rgba(245, 158, 11, 0.13);
            color: #b45309;
        }

        .callout {
            border-left: 4px solid #2563eb;
            padding: 0.85rem 1rem;
            background: rgba(37, 99, 235, 0.06);
            border-radius: 10px;
            margin: 0.5rem 0 1rem 0;
            color: #1f2937;
        }

        .callout-success {
            border-left-color: #059669;
            background: rgba(5, 150, 105, 0.07);
        }

        .callout-warn {
            border-left-color: #d97706;
            background: rgba(217, 119, 6, 0.08);
        }

        .small-note {
            font-size: 0.92rem;
            color: #6b7280;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def csv_or_none(filename: str) -> Optional[pd.DataFrame]:
    path = REPORTS / filename
    if not path.exists():
        return None

    df = pd.read_csv(path, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]

    for col in ["strategy", "asset", "market", "venue", "episode", "day", "date", "mode"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    return df


def show_df(df: Optional[pd.DataFrame], digits: int = 4) -> None:
    if df is None:
        st.info("Table not available.")
        return
    st.dataframe(df.round(digits), hide_index=True, use_container_width=True)


def doc_text_or_none(filename: str) -> Optional[str]:
    path = DOCS / filename
    if path.exists():
        return path.read_text(encoding="utf-8")
    return None


def report_text_or_none(filename: str) -> Optional[str]:
    path = REPORTS / filename
    if path.exists():
        return path.read_text(encoding="utf-8")
    return None


def show_image(filename: str, caption: str) -> None:
    path = REPORTS / filename
    if path.exists():
        try:
            st.image(str(path), caption=caption, use_container_width=True)
        except TypeError:
            # Compatibility fallback for older Streamlit versions
            st.image(str(path), caption=caption)
    else:
        st.info(f"Missing image: reports/{filename}")


def show_df(df: Optional[pd.DataFrame], digits: int = 4) -> None:
    if df is None:
        st.info("Table not available.")
        return
    st.dataframe(df.round(digits), hide_index=True)


def get_row(df: Optional[pd.DataFrame], strategy: str) -> Optional[pd.Series]:
    if df is None or "strategy" not in df.columns:
        return None
    sub = df.loc[df["strategy"] == strategy]
    if len(sub) == 0:
        return None
    return sub.iloc[0]


def get_asset_row(df: Optional[pd.DataFrame], asset: str, strategy: str) -> Optional[pd.Series]:
    if df is None:
        return None
    if not {"asset", "strategy"}.issubset(df.columns):
        return None
    sub = df.loc[(df["asset"] == asset) & (df["strategy"] == strategy)]
    if len(sub) == 0:
        return None
    return sub.iloc[0]


def metric_value(row: Optional[pd.Series], key: str, fmt: str = ".4f", default: str = "N/A") -> str:
    if row is None or key not in row:
        return default
    try:
        return format(float(row[key]), fmt)
    except Exception:
        return str(row[key])


def int_metric_value(row: Optional[pd.Series], key: str, default: str = "N/A") -> str:
    if row is None or key not in row:
        return default
    try:
        return str(int(row[key]))
    except Exception:
        return str(row[key])


def badge(label: str, kind: str = "real") -> str:
    css = {
        "real": "badge-real",
        "synth": "badge-synth",
        "external": "badge-external",
        "method": "badge-method",
    }.get(kind, "badge-real")
    return f'<span class="badge {css}">{label}</span>'


def hero(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="hero">
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_title(text: str) -> None:
    st.markdown(f'<div class="section-title">{text}</div>', unsafe_allow_html=True)


def phase_card(title: str, badges_html: str, question: str, baselines: str, ours: str, evidence: str) -> None:
    st.markdown(
        f"""
        <div class="phase-card">
            <div>{badges_html}</div>
            <h3>{title}</h3>
            <p><b>Question.</b> {question}</p>
            <p><b>Baselines.</b> {baselines}</p>
            <p><b>Method.</b> {ours}</p>
            <p><b>What it shows.</b> {evidence}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def callout(text: str, tone: str = "info") -> None:
    extra = ""
    if tone == "success":
        extra = " callout-success"
    elif tone == "warn":
        extra = " callout-warn"
    st.markdown(f'<div class="callout{extra}">{text}</div>', unsafe_allow_html=True)
    