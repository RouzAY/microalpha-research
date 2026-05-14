# app/pages/5_Phase_2_Appendix.py
import sys
from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.append(str(APP_DIR))

from common import (
    badge,
    callout,
    csv_or_none,
    get_row,
    hero,
    inject_css,
    metric_value,
    report_text_or_none,
    section_title,
    show_df,
    show_image,
)

st.set_page_config(page_title="Phase 2 — Appendix", layout="wide")
inject_css()

summary2 = csv_or_none("alpha_phase2_summary.csv")
ramp = get_row(summary2, "ramp")

hero(
    "Phase 2 — Appendix / Research Sandbox",
    "Synthetic multi-symbol environment calibrated from real microstructure features, used as an intermediate sandbox between execution research and fully real historical alpha evaluation.",
)

st.markdown(
    badge("synthetic", "synth") + badge("calibrated", "method") + badge("appendix", "method"),
    unsafe_allow_html=True,
)

callout(
    """
<b>What this page shows.</b><br>
Phase 2 is a controlled research sandbox.
It is designed to test multi-horizon, cross-sectional, and regime-adaptive ideas
in an environment that is synthetic but still calibrated from real microstructure features.<br><br>
It is therefore useful for experimentation, but it is intentionally separated from the stronger real-data evidence in Phases 3 and 4.
""",
    tone="info",
)

# m1, m2 = st.columns(2)
# m1.metric("RAMP mean net bps / step", metric_value(ramp, "mean_net_bps", ".4f"))
# m2.metric("RAMP turnover", metric_value(ramp, "turnover", ".4f"))

m1, m2, m3 = st.columns(3)
m1.metric("RAMP mean net bps / step", metric_value(ramp, "mean_net_bps", ".4f"))
m2.metric("RAMP turnover", metric_value(ramp, "mean_turnover", ".4f"))
m3.metric("RAMP hit rate", metric_value(ramp, "hit_rate", ".4f"))

section_title("How to read this phase")

c1, c2 = st.columns(2)

with c1:
    st.markdown(
        """
**Synthetic but calibrated**  
The environment is not fully real, but it is built from real microstructure-derived ingredients.

**Why this matters**  
It allows controlled testing of ideas that would be harder to isolate directly in a full real-data benchmark.
"""
    )

with c2:
    st.markdown(
        """
**What this phase is not**  
This is not the strongest evidence in the project.

**What it is for**  
It is a sandbox for:
- multi-horizon prediction,
- cross-sectional ranking,
- cost-aware allocation,
- regime adaptation.
"""
    )

callout(
    """
<b>Interpretation.</b><br>
Phase 2 is best understood as an intermediate laboratory:
more realistic than an arbitrary toy simulation,
but not as strong as the fully real benchmarks used later.
""",
    tone="success",
)

section_title("Methods compared in this phase")

l1, l2 = st.columns(2)

with l1:
    st.subheader("Baseline strategies")
    st.markdown(
        """
- **uniform**  
  Equal treatment across symbols

- **imbalance**  
  Uses imbalance-style ranking

- **momentum**  
  Uses simple return continuation logic

- **ridge5**  
  Linear predictive baseline
"""
    )

with l2:
    st.subheader("Method: RAMP")
    st.markdown(
        """
**RAMP = Regime-Adaptive Multi-horizon Portfolio**

It combines:
- multiple prediction horizons,
- regime-dependent weighting,
- cost-aware normalization,
- and market-neutral cross-sectional portfolio construction.
"""
    )

section_title("Why this phase exists")

st.markdown(
    """
The reason to include this phase is methodological.

Before testing everything directly in a real-data benchmark, it is useful to study:
- whether multiple horizons should be combined,
- whether costs should shrink signals,
- whether regime adaptation changes ranking quality,
- and whether a market-neutral portfolio rule behaves sensibly.

That is exactly the role of Phase 2.
"""
)

callout(
    """
<b>Key point.</b><br>
Phase 2 is not the final destination of the project.
It is a bridge between:
<b>execution research</b> and <b>fully real directional alpha evaluation</b>.
""",
    tone="info",
)

section_title("How to interpret the main metrics")

c3, c4 = st.columns(2)

with c3:
    st.markdown(
        """
**Mean net bps / step**  
Average gain per decision step after costs.

**Turnover**  
How much the allocation changes over time.
High turnover may indicate more fragile or more expensive behavior.
"""
    )

with c4:
    st.markdown(
        """
Because this phase is cross-sectional and synthetic-calibrated,
the metrics should be interpreted mainly as:
- evidence of controlled signal quality,
- evidence of allocation quality,
- evidence of cost-aware portfolio behavior.
"""
    )

section_title("Benchmark table")
show_df(summary2)

callout(
    """
<b>How to read the table.</b><br>
Compare methods first by <b>mean net bps / step</b>, then check whether that performance is obtained with reasonable turnover.
The goal is not only to find a stronger predictor, but a stronger and more coherent portfolio rule.
""",
    tone="success",
)

section_title("Visual summaries")

r1, r2 = st.columns(2)
with r1:
    show_image("alpha_phase2_mean_net_bps.png", "Phase 2 — mean net bps / step")
with r2:
    show_image("alpha_phase2_cumulative_net_bps.png", "Phase 2 — cumulative net bps")

show_image("alpha_phase2_episode_boxplot.png", "Phase 2 — episode-level distribution")

callout(
    """
<b>How to read the plots.</b><br>
The mean-net-bps plot compares average strategy quality.
The cumulative plot shows how gains accumulate through the benchmark.
The episode-level distribution helps assess whether performance is broad-based or concentrated in a few episodes.
""",
    tone="info",
)

section_title("Why this phase is placed in the appendix")

st.markdown(
    """
This phase is placed in the appendix because the strongest evidence of the project comes later:

- **Phase 3** provides fully real historical walk-forward alpha,
- **Phase 4** adds robustness under cross-market shift.

So Phase 2 remains useful and informative,
but it is intentionally positioned as supporting evidence rather than the main conclusion.
"""
)

summary_md = report_text_or_none("alpha_phase2_summary.md")
with st.expander("Open written phase-2 summary"):
    if summary_md is None:
        st.info("alpha_phase2_summary.md not found in reports/")
    else:
        st.markdown(summary_md)

section_title("Takeaway")

callout(
    """
<b>Main takeaway.</b><br>
Phase 2 provides a controlled environment for testing alpha construction ideas
before moving to harder real-data benchmarks.<br><br>
Its role is to make the transition from execution research to real directional alpha
more structured and easier to interpret.
""",
    tone="success",
)
