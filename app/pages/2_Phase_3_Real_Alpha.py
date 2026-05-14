# app/pages/2_Phase_3_Real_Alpha.py
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

st.set_page_config(page_title="Phase 3 — Real Alpha", layout="wide")
inject_css()

summary3 = csv_or_none("alpha_phase3_summary.csv")
held3 = csv_or_none("alpha_phase3_heldout_summary.csv")
ramp = get_row(summary3, "ramp_real")

hero(
    "Phase 3 — Real Walk-Forward Alpha",
    "Fully real historical evaluation on AAPL order-book days with explicit spread-based costs and anchored walk-forward retraining.",
)

st.markdown(
    badge("real", "real") + badge("walk-forward", "method") + badge("historical alpha", "method"),
    unsafe_allow_html=True,
)

callout(
    """
<b>What this page shows.</b><br>
This phase studies whether microstructure-aware signals remain useful on <b>real historical data</b>,
after explicit transaction costs, in a strict <b>walk-forward</b> protocol.<br><br>
This is the first phase where the project moves from a controlled research setup to a fully real directional-alpha benchmark.
""",
    tone="info",
)

m1, m2, m3, m4 = st.columns(4)
m1.metric("RAMP-R mean net bps / step", metric_value(ramp, "mean_net_bps", ".4f"))
m2.metric("RAMP-R total net bps", metric_value(ramp, "total_net_bps", ".1f"))
m3.metric("RAMP-R turnover", metric_value(ramp, "turnover", ".4f"))
m4.metric("RAMP-R hit rate", metric_value(ramp, "hit_rate", ".4f"))

section_title("How to read these numbers")

c1, c2 = st.columns(2)

with c1:
    st.markdown(
        """
**Mean net bps / step**  
Average gain per decision step after transaction costs.

**Total net bps**  
Cumulative benchmark gain across all evaluated steps.

**Turnover**  
How much the strategy changes position over time.
High turnover can indicate higher cost burden.
"""
    )

with c2:
    st.markdown(
        """
**Hit rate**  
Fraction of realized outcomes that are positive.

**Why these metrics matter**  
This phase is not about classification accuracy.
It is about whether the signal remains economically useful:
- after costs,
- across days,
- in a walk-forward protocol.
"""
    )

callout(
    """
<b>Interpretation.</b><br>
A positive <b>mean net bps / step</b> indicates that, on average, the strategy adds value after cost.
A positive <b>total net bps</b> indicates cumulative benchmark value.
Turnover helps assess whether performance comes from sensible trading activity or excessive churn.
""",
    tone="success",
)

section_title("Methods compared in this phase")

left, center, right = st.columns(3)

with left:
    st.subheader("Baseline strategies")
    st.markdown(
        """
- **momentum**  
  Uses recent return as a simple directional predictor

- **signed_flow**  
  Uses aggressive trade-flow imbalance

- **micro_fixed**  
  Uses microprice pressure with fixed sign

- **ridge_static**  
  Fixed linear predictor

- **online_regime**  
  Simpler adaptive microstructure rule
"""
    )

with center:
    st.subheader("Method: RAMP-R")
    st.markdown(
        """
**RAMP-R = Regime-Adaptive Microstructure Portfolio on Real data**

RAMP-R combines:
- a microstructure signal,
- a predictor fit on prior days only,
- warmup-based sign adaptation,
- thresholding so that the strategy trades only when conviction is high enough,
- explicit spread-based transaction costs.
"""
    )

with right:
    st.subheader("Why this phase matters")
    st.markdown(
        """
Phase 1 answers an execution question.

Phase 3 answers a different question:

> can short-horizon order-book information generate real directional alpha after cost?

That is why the baselines here are no longer TWAP or VWAP.
This phase is about **directional decision-making**, not order scheduling.
"""
    )

section_title("Walk-forward protocol")

st.markdown(
    """
The benchmark uses an anchored walk-forward setup:

1. train on historical days strictly before the test day,  
2. evaluate on the next real day,  
3. repeat through the benchmark horizon.

This avoids using future information and makes the protocol much more realistic than an in-sample fit.
"""
)

callout(
    """
<b>Core point.</b><br>
The model used on a given day is trained only on earlier days.
So the benchmark is a genuine out-of-sample historical protocol.
""",
    tone="success",
)

section_title("Signal construction")

st.markdown(
    r"""
The future return target at horizon \(H\) is
"""
)

st.latex(
    r"""
y_t = \left(\frac{m_{t+H}}{m_t}-1\right)\times 10^4
"""
)

st.markdown(
    r"""
where \(m_t\) is the midprice.
So \(y_t\) is the forward return in basis points.
"""
)

st.markdown(
    """
The hybrid signal used by RAMP-R is
"""
)

st.latex(
    r"""
s_t^{\mathrm{RAMP-R}}
=
\omega_\mu\,\mathrm{sig\_micro}_t
+
\omega_{\mathrm{ML}}\,\tilde f_t
"""
)

st.markdown(
    r"""
where:
- \(\mathrm{sig\_micro}_t\) is a microstructure signal derived from the order book,
- \(\tilde f_t\) is a normalized ML prediction estimated from prior days.
"""
)

st.markdown(
    """
A warmup statistic is also used to adapt the sign of the strategy when the local intraday regime appears inverted:
"""
)

st.latex(
    r"""
\beta_{\mathrm{warm}}
=
\frac{\mathrm{Cov}(\mathrm{sig\_micro}_{1:W},y_{1:W})}
{\mathrm{Var}(\mathrm{sig\_micro}_{1:W})+\varepsilon}
"""
)

callout(
    """
<b>Interpretation.</b><br>
If the warmup period suggests that the local relation between the microstructure signal and returns
is different from the historical average, the strategy can adapt its sign.
""",
    tone="info",
)

section_title("Thresholding and cost model")

st.markdown(
    """
The strategy does not trade all the time.
It first estimates a threshold on the warmup period and trades only when the signal is strong enough.
"""
)

st.latex(
    r"""
\theta = \mathrm{Quantile}_q\big(|s_1|,\dots,|s_W|\big)
"""
)

st.latex(
    r"""
p_t
=
\begin{cases}
0, & |s_t| < \theta,\\
\mathrm{sign}(\sigma s_t), & |s_t|\ge \theta.
\end{cases}
"""
)

st.markdown(
    """
This means:
- stay flat when conviction is low,
- take directional exposure only when the signal is sufficiently strong.
"""
)

st.latex(
    r"""
n_t = g_t - c_t,
\qquad
g_t = p_t y_t,
\qquad
c_t = \lambda_{\mathrm{cost}}\mathrm{spr}^{\mathrm{bps}}_t |p_t-p_{t-1}|
"""
)

st.markdown(
    r"""
So:
- \(g_t\) is gross P\&L,
- \(c_t\) is the transaction-cost penalty,
- \(n_t\) is net P\&L, the economically relevant quantity.
"""
)

section_title("What this phase proves")

st.markdown(
    """
This phase does **not** try to prove that short-horizon prediction is easy.

Instead, it demonstrates the following:

1. real historical microstructure contains usable directional information;  
2. that information can survive explicit spread-based cost penalties;  
3. warmup adaptation helps reduce the risk of using the wrong local sign;  
4. hybrid microstructure + ML logic can outperform simpler baselines.
"""
)

section_title("Benchmark tables")

tabs = st.tabs(["Combined summary", "Strict held-out block"])

with tabs[0]:
    st.subheader("Phase-3 combined summary")
    show_df(summary3)

    st.markdown(
        """
Typical reading:
- compare methods by **mean net bps / step**,
- inspect **total net bps** for cumulative gain,
- check **turnover** and **hit rate** for behavioral profile,
- look at **positive episodes** and **episode t-stat** for stability.
"""
    )

with tabs[1]:
    st.subheader("Strict held-out final block")
    show_df(held3)

    st.markdown(
        """
This held-out block is useful because it isolates a strict final segment of the benchmark.
It helps answer whether performance remains visible on a particularly clean out-of-sample slice.
"""
    )

section_title("Visual summaries")

r1, r2 = st.columns(2)
with r1:
    show_image("alpha_phase3_mean_net_bps.png", "Phase 3 — mean net bps / step")
with r2:
    show_image("alpha_phase3_cumulative_net_bps.png", "Phase 3 — cumulative net bps")

show_image("alpha_phase3_daily_total_net_bps.png", "Phase 3 — daily total net bps")
show_image("alpha_phase3_heldout_mean_net_bps.png", "Phase 3 — held-out mean net bps / step")

callout(
    """
<b>How to read the plots.</b><br>
The mean-net-bps plot compares average strategy quality.<br>
The cumulative-net-bps plot shows how gains accumulate through the benchmark.<br>
The daily-total plot helps assess whether performance is spread across days or concentrated in a few periods.
""",
    tone="info",
)

section_title("Written summary")

summary_md = report_text_or_none("alpha_phase3_summary.md")
if summary_md is None:
    st.info("alpha_phase3_summary.md not found in reports/")
else:
    with st.expander("Open written phase-3 summary"):
        st.markdown(summary_md)

section_title("Takeaway")

callout(
    """
<b>Main takeaway.</b><br>
Phase 3 shows that the project is not limited to synthetic experiments or execution-only logic.
It demonstrates a fully real historical directional-alpha benchmark with explicit costs,
walk-forward retraining, and adaptive microstructure-based decision-making.
""",
    tone="success",
)
