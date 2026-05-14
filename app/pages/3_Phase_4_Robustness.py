# app/pages/3_Phase_4_Robustness.py
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.append(str(APP_DIR))

from common import (
    badge,
    callout,
    csv_or_none,
    get_asset_row,
    get_row,
    hero,
    inject_css,
    metric_value,
    report_text_or_none,
    section_title,
    show_df,
    show_image,
)

st.set_page_config(page_title="Phase 4 — Robustness", layout="wide")
inject_css()

summary4 = csv_or_none("alpha_phase4_summary.csv")
asset4 = csv_or_none("alpha_phase4_asset_summary.csv")

# Optional detailed files if they exist
episode4 = csv_or_none("alpha_phase4_episode_summary.csv")
results4 = csv_or_none("alpha_phase4_results.csv")

camp = get_row(summary4, "camp_r")
aapl = get_asset_row(asset4, "AAPL", "camp_r")
bnb = get_asset_row(asset4, "BNBUSDT", "camp_r")

hero(
    "Phase 4 — Cross-Asset / Cross-Venue Robustness",
    "Real AAPL walk-forward evaluation plus an external Binance transfer test under an explicit domain-shift gate.",
)

st.markdown(
    badge("real", "real") + badge("external", "external") + badge("drift-aware", "method"),
    unsafe_allow_html=True,
)

callout(
    """
<b>What this page shows.</b><br>
This phase evaluates whether the strategy remains useful when the target market differs from the
market used for training and calibration.<br><br>
<b>AAPL</b> is the primary real-data market used in the main historical benchmark.<br>
<b>BNBUSDT</b> is an external market used as a transfer test.<br><br>
The role of <b>CAMP-R</b> is not simply to maximize raw predictive strength.
Its role is to <b>adjust model trust under shift</b>:
use the stronger hybrid model when the episode looks familiar, and use a safer fallback when it does not.
""",
    tone="info",
)

m1, m2, m3, m4 = st.columns(4)
m1.metric("CAMP-R mean net bps / step", metric_value(camp, "mean_net_bps", ".4f"))
m2.metric("Total net bps", metric_value(camp, "total_net_bps", ".1f"))
m3.metric("AAPL mean net bps / step", metric_value(aapl, "mean_net_bps", ".4f"))
m4.metric("BNBUSDT mean net bps / step", metric_value(bnb, "mean_net_bps", ".4f"))

section_title("How to read these numbers")

c1, c2 = st.columns(2)

with c1:
    st.markdown(
        """
**Mean net bps / step**  
Average gain per decision step after transaction costs.

**Total net bps**  
Cumulative benchmark gain across all evaluated steps and episodes.

**AAPL mean net bps / step**  
Performance on the main real-data market, i.e. the primary in-domain benchmark.
"""
    )

with c2:
    st.markdown(
        """
**BNBUSDT mean net bps / step**  
Performance on an external market used as a transfer test.

A lower value on BNBUSDT than on AAPL does **not** necessarily mean failure.  
It often means:
- the market is structurally different,
- transfer is harder,
- the fallback mechanism is preventing a collapse.
"""
    )

callout(
    """
<b>Interpretation.</b><br>
A strong value on <b>AAPL</b> shows that the hybrid model is effective on the main benchmark market.<br>
A still-positive value on <b>BNBUSDT</b> suggests that the strategy remains usable under transfer,
even though the external market is different.
""",
    tone="success",
)

section_title("Methods compared in this phase")

left, center, right = st.columns(3)

with left:
    st.subheader("Baseline strategies")
    st.markdown(
        """
- **ridge_static**  
  Fixed linear predictor with no explicit shift control

- **online_regime**  
  Simpler adaptive microstructure rule

- **RAMP-R**  
  Hybrid real-data model without the explicit cross-venue shift gate
"""
    )

with center:
    st.subheader("Method: CAMP-R")
    st.markdown(
        """
**CAMP-R = Cross-Asset Adaptive Microstructure Portfolio on Real data**

CAMP-R combines:
- a stronger hybrid model when the current episode looks close to training,
- a safer adaptive fallback when the current episode looks shifted.

So CAMP-R is not only a predictive model.
It is a **decision rule about model trust**.
"""
    )

with right:
    st.subheader("What changes from Phase 3")
    st.markdown(
        """
In Phase 3, the project studies real walk-forward alpha on the main market.

In Phase 4, the project adds a new question:

> should the same model be trusted to the same degree when the target market changes?

That is why this phase is about **robustness** and **model-risk control**.
"""
    )

section_title("Shift-aware decision rule")

st.markdown(
    """
The strategy first measures how different the current episode looks relative to the training distribution.
This is done during a warmup segment of the episode.
"""
)

st.latex(
    r"""
D_{\mathrm{shift}}
=
\sqrt{
\frac{1}{Wd}
\sum_{t=1}^{W}\sum_{j=1}^{d}
\left(
\frac{x_{t,j}-\mu_{\mathrm{train},j}}
{\sigma_{\mathrm{train},j}}
\right)^2
}
"""
)

st.markdown(
    r"""
Here:
- \(W\) is the warmup length,
- \(d\) is the number of features,
- \(x_{t,j}\) is the current feature value,
- \(\mu_{\mathrm{train},j}\) and \(\sigma_{\mathrm{train},j}\) are the training mean and standard deviation.

So \(D_{\mathrm{shift}}\) is a standardized distance between the current episode and the training domain.
"""
)

st.latex(
    r"""
\pi_{\mathrm{CAMP-R}}
=
\begin{cases}
\pi_{\mathrm{online\_regime}}, & D_{\mathrm{shift}}>\tau_{\mathrm{shift}},\\
\pi_{\mathrm{RAMP-R}}, & D_{\mathrm{shift}}\le \tau_{\mathrm{shift}}.
\end{cases}
"""
)

c3, c4 = st.columns(2)

with c3:
    callout(
        """
<b>If shift is low.</b><br>
The current episode looks close to training.<br>
CAMP-R uses <b>RAMP-R</b>, the stronger hybrid model.
""",
        tone="success",
    )

with c4:
    callout(
        """
<b>If shift is high.</b><br>
The current episode looks different from training.<br>
CAMP-R reduces model trust and uses <b>online_regime</b>, a simpler fallback.
""",
        tone="warn",
    )

section_title("What this phase proves")

st.markdown(
    """
This phase does **not** claim that every market should produce the same performance.

Instead, it demonstrates a more realistic principle:

1. the hybrid model is strongest in-domain;
2. transfer to a new market is harder;
3. explicit shift detection helps avoid over-trusting the learned model;
4. a fallback rule can preserve usefulness under market change.
"""
)

section_title("Benchmark tables")

tabs = st.tabs(["Combined summary", "Per-asset summary", "Optional episode view"])

with tabs[0]:
    st.subheader("Combined phase-4 summary")
    show_df(summary4)

    st.markdown(
        """
Typical reading:
- compare the ranking of methods by **mean net bps / step**,
- check whether higher performance is obtained with reasonable **turnover** and **cost**,
- look at **positive episodes** and **episode t-stat** for stability.
"""
    )

with tabs[1]:
    st.subheader("Per-asset summary")
    show_df(asset4)

    st.markdown(
        """
This table helps separate:
- **in-domain behavior** on AAPL,
- **transfer behavior** on BNBUSDT.

That separation is essential for understanding whether the final method is simply strong in-domain
or also robust under external shift.
"""
    )

with tabs[2]:
    st.subheader("Episode-by-episode view")
    if episode4 is not None:
        show_df(episode4)
        st.markdown(
            """
This table can be used to inspect whether performance is concentrated in only a few episodes
or is more broadly distributed across the benchmark.
"""
        )
    elif results4 is not None:
        cols = [c for c in results4.columns if c in [
            "asset", "episode", "day", "strategy", "net_bps", "gross_bps", "cost_bps",
            "turnover", "mode", "shift_score", "is_shifted"
        ]]
        if len(cols) > 0:
            preview = results4[cols].copy()
            show_df(preview.head(200))
            st.info(
                "A dedicated alpha_phase4_episode_summary.csv file was not found, so a preview "
                "from alpha_phase4_results.csv is shown instead."
            )
        else:
            st.info("No detailed episode-level columns were found in alpha_phase4_results.csv.")
    else:
        st.info(
            "No detailed episode file was found. "
            "If you want, you can later generate an alpha_phase4_episode_summary.csv for a clearer episode-by-episode view."
        )

section_title("Visual summaries")

v1, v2 = st.columns(2)
with v1:
    show_image("alpha_phase4_mean_net_bps.png", "Phase 4 — mean net bps / step")
with v2:
    show_image("alpha_phase4_asset_mean_net_bps.png", "Phase 4 — per-asset mean net bps / step")

v3, v4 = st.columns(2)
with v3:
    show_image("alpha_phase4_cumulative_net_bps.png", "Phase 4 — cumulative net bps")
with v4:
    show_image("alpha_phase4_episode_total_net_bps.png", "Phase 4 — episode-level total net bps")

show_image("alpha_phase4_external_mean_net_bps.png", "Phase 4 — external market mean net bps / step")

callout(
    """
<b>How to read the plots.</b><br>
The mean-net-bps plots compare average strategy quality.<br>
The cumulative-net-bps plot shows how performance accumulates through the benchmark.<br>
The episode-level plot helps assess whether the result is broad-based or concentrated in a few episodes.
""",
    tone="info",
)

section_title("Written summary")

summary_md = report_text_or_none("alpha_phase4_summary.md")
if summary_md is None:
    st.info("alpha_phase4_summary.md not found in reports/")
else:
    with st.expander("Open written phase-4 summary"):
        st.markdown(summary_md)

section_title("Takeaway")

callout(
    """
<b>Main takeaway.</b><br>
Phase 4 shows that robustness is not only about building a stronger predictor.
It is also about deciding <b>when the predictor should be trusted</b>.<br><br>
CAMP-R is designed around that principle:
<b>strong hybrid behavior in familiar conditions, safer fallback behavior under shift.</b>
""",
    tone="success",
)
