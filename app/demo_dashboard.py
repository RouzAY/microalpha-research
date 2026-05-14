# app/demo_dashboard.py
import streamlit as st

from common import (
    badge,
    callout,
    csv_or_none,
    get_asset_row,
    get_row,
    hero,
    inject_css,
    int_metric_value,
    metric_value,
    phase_card,
    section_title,
    show_df,
    show_image,
)

st.set_page_config(page_title="MicroAlpha Dashboard", layout="wide")
inject_css()

summary4 = csv_or_none("alpha_phase4_summary.csv")
asset4 = csv_or_none("alpha_phase4_asset_summary.csv")
held3 = csv_or_none("alpha_phase3_heldout_summary.csv")
summary1 = csv_or_none("alpha_phase1_summary.csv")  # optional, if you later add one
summary2 = csv_or_none("alpha_phase2_summary.csv")
summary3 = csv_or_none("alpha_phase3_summary.csv")

camp = get_row(summary4, "camp_r")
aapl = get_asset_row(asset4, "AAPL", "camp_r")
bnb = get_asset_row(asset4, "BNBUSDT", "camp_r")

hero(
    "MicroAlpha Research Dashboard",
    "A microstructure-aware platform for execution research, real-data alpha evaluation, and cross-venue robustness under explicit transaction costs.",
)

st.sidebar.markdown("## Navigation")
st.sidebar.markdown(
    """
- Overview  
- Phase 1 — Execution  
- Phase 3 — Real Alpha  
- Phase 4 — Robustness  
- Methods & Limitations  
- Phase 2 — Appendix
"""
)

callout(
    """
<b>How to use this dashboard.</b><br>
This dashboard is organized in increasing order of difficulty:
<b>execution</b> first, then a <b>controlled alpha sandbox</b>, then <b>real historical alpha</b>,
and finally <b>robustness under market shift</b>.<br><br>
A good reading order is:
<ol>
<li><b>Phase 1</b> to understand execution and the comparison against TWAP/VWAP,</li>
<li><b>Phase 3</b> to see real walk-forward directional alpha,</li>
<li><b>Phase 4</b> to understand robustness and model-trust control,</li>
<li><b>Methods & Limitations</b> to understand exactly what is and is not modeled.</li>
</ol>
""",
    tone="info",
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("CAMP-R mean net bps / step", metric_value(camp, "mean_net_bps", ".4f"))
c2.metric("Total net bps", metric_value(camp, "total_net_bps", ".1f"))
c3.metric(
    "Positive episodes",
    f"{int_metric_value(camp, 'positive_episodes')}/{int_metric_value(camp, 'n_episodes')}",
)
c4.metric("Episode t-stat", metric_value(camp, "tstat_episode_total", ".2f"))

c5, c6 = st.columns(2)
c5.metric("AAPL mean net bps / step", metric_value(aapl, "mean_net_bps", ".4f"))
c6.metric("BNBUSDT mean net bps / step", metric_value(bnb, "mean_net_bps", ".4f"))

section_title("What this project studies")

st.markdown(
    """
This project answers three distinct questions:

1. **Execution:** if a parent order is already known, can it be executed better than standard schedules such as **TWAP** and **VWAP**?  
2. **Directional alpha:** can short-horizon order-book information produce useful signals **after costs** on real historical data?  
3. **Robustness:** if the market changes, should the strategy trust the same predictive model to the same degree?

These are different problems, which is why the project is split into phases.
"""
)

section_title("Project roadmap")

st.markdown(
    """
### Phase progression

**Phase 1 → Phase 2 → Phase 3 → Phase 4**

- **Phase 1:** execution benchmark  
- **Phase 2:** synthetic but real-calibrated alpha sandbox  
- **Phase 3:** fully real walk-forward directional alpha  
- **Phase 4:** cross-asset / cross-venue robustness under shift
"""
)

p1, p2, p3, p4 = st.columns(4)

with p1:
    phase_card(
        "Phase 1 — Execution",
        badge("real", "real") + badge("execution", "method"),
        "Given a parent order, can child orders be scheduled better than simple execution baselines?",
        "TWAP, VWAP, Liquidity-only, Alpha-only",
        "MALP",
        "Microstructure information improves execution quality under explicit liquidity costs.",
    )

with p2:
    phase_card(
        "Phase 2 — Sandbox",
        badge("synthetic", "synth") + badge("calibrated", "method"),
        "Can cross-sectional and multi-horizon alpha ideas be tested in a controlled setting?",
        "uniform, imbalance, momentum, ridge5",
        "RAMP",
        "Provides a controlled environment for testing alpha construction before fully real evaluation.",
    )

with p3:
    phase_card(
        "Phase 3 — Real Alpha",
        badge("real", "real") + badge("walk-forward", "method"),
        "Can microstructure-aware signals remain positive after costs on real historical data?",
        "momentum, signed_flow, micro_fixed, ridge_static, online_regime",
        "RAMP-R",
        "Shows real directional alpha under explicit spread-based transaction costs.",
    )

with p4:
    phase_card(
        "Phase 4 — Robustness",
        badge("real", "real") + badge("external", "external"),
        "What happens when the target market differs from the training market?",
        "ridge_static, online_regime, RAMP-R",
        "CAMP-R",
        "Adds model-trust control under shift by switching between a stronger hybrid model and a safer fallback.",
    )

section_title("What each phase answers")

q1, q2 = st.columns(2)

with q1:
    st.markdown(
        """
### Execution vs alpha

**Phase 1 is an execution problem**
- the parent order is already known,
- the question is how to trade it,
- so the correct baselines are **TWAP** and **VWAP**.

**Phases 3 and 4 are alpha problems**
- the question is whether to take directional exposure,
- and how to handle costs and drift,
- so TWAP and VWAP are no longer the right comparators.
"""
    )

with q2:
    st.markdown(
        """
### Why the phases are separated

The project deliberately avoids mixing:
- **execution quality**,  
- **directional signal quality**,  
- **robustness under shift**.  

Separating them makes the benchmark easier to interpret:
each phase answers one main question with the right baselines.
"""
    )

callout(
    """
<b>Key idea.</b><br>
The dashboard should not be read as four unrelated experiments.
It should be read as a progression:
<b>execute better → test alpha safely → validate on real data → control model risk under shift.</b>
""",
    tone="success",
)

section_title("How to interpret the headline metrics")

mleft, mright = st.columns(2)

with mleft:
    st.markdown(
        """
**Implementation shortfall (Phase 1)**  
Execution metric. Lower is better.

**Mean net bps / step (Phases 2–4)**  
Average gain per decision step after costs.

**Total net bps**  
Cumulative gain over the benchmark.
"""
    )

with mright:
    st.markdown(
        """
**Turnover**  
How much the position changes over time. High turnover can imply higher cost burden.

**Positive episodes**  
How many benchmark episodes end positive.

**Episode t-stat**  
Rough measure of stability across episodes.
"""
    )

section_title("Main benchmark snapshot")

left, right = st.columns([1.1, 1.0])

with left:
    st.subheader("Phase 4 combined summary")
    show_df(summary4)
    show_image("alpha_phase4_mean_net_bps.png", "Phase 4 — mean net bps / step")

with right:
    st.subheader("Phase 4 per-asset view")
    show_df(asset4)
    show_image("alpha_phase4_asset_mean_net_bps.png", "Phase 4 — per-asset mean net bps / step")

section_title("Real-data checkpoint")

st.markdown(
    """
The strongest real-data evidence in the project comes from:
- **Phase 3:** real historical walk-forward alpha on AAPL,
- **Phase 4:** real AAPL plus external transfer on BNBUSDT.
"""
)

r1, r2 = st.columns(2)
with r1:
    show_image("alpha_phase3_cumulative_net_bps.png", "Phase 3 — cumulative net bps")
with r2:
    show_image("alpha_phase4_cumulative_net_bps.png", "Phase 4 — cumulative net bps")

st.subheader("Strict held-out final block from Phase 3")
show_df(held3)
show_image("alpha_phase3_heldout_mean_net_bps.png", "Phase 3 — held-out mean net bps / step")

section_title("What is modeled and what is not")

l1, l2 = st.columns(2)

with l1:
    st.markdown(
        """
### Modeled explicitly
- spread and displayed top-of-book liquidity
- level-1 / level-2 imbalance
- signed trade flow
- explicit spread-based transaction costs
- walk-forward retraining
- warmup adaptation
- domain-shift fallback
"""
    )

with l2:
    st.markdown(
        """
### Not modeled explicitly
- exact queue position
- passive fill probability in the queue
- detailed cancellation / replacement dynamics
- latency-sensitive exchange behavior
- full production-grade order management
"""
    )

callout(
    """
<b>Scope.</b><br>
The correct interpretation of the project is:
<b>a research-grade microstructure benchmark and alpha-validation platform</b>,
not a full exchange simulator.
""",
    tone="warn",
)

section_title("How to navigate next")

st.markdown(
    """
- Go to **Phase 1** to understand execution and the comparison against **TWAP / VWAP**  
- Go to **Phase 3** to understand real historical directional alpha  
- Go to **Phase 4** to understand robustness under market shift  
- Go to **Methods & Limitations** to understand the modeling scope  
- Go to **Phase 2** for the controlled research sandbox
"""
)