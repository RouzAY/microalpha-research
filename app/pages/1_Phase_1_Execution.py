# app/pages/1_Phase_1_Execution.py
import sys
from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.append(str(APP_DIR))

from common import (
    badge,
    callout,
    hero,
    inject_css,
    report_text_or_none,
    section_title,
    show_image,
)

st.set_page_config(page_title="Phase 1 — Execution", layout="wide")
inject_css()

hero(
    "Phase 1 — Execution Research",
    "Parent-order scheduling under top-of-book liquidity constraints, with explicit comparison against standard execution schedules such as TWAP and VWAP.",
)

st.markdown(
    badge("real", "real") + badge("execution", "method") + badge("TWAP/VWAP benchmark", "method"),
    unsafe_allow_html=True,
)

callout(
    """
<b>What this page shows.</b><br>
This phase studies the <b>execution problem</b>: once a trader already knows that they want to buy or sell,
how should the order be split across time?<br><br>
This is the correct phase to compare against <b>TWAP</b> and <b>VWAP</b>,
because those are execution schedules for a known parent order.
""",
    tone="info",
)

section_title("How to read this phase")

c1, c2 = st.columns(2)

with c1:
    st.markdown(
        """
**Parent order**  
A large order that must be executed over a finite horizon.

**Child orders**  
Smaller pieces of the parent order sent sequentially over time.

**Implementation shortfall (bps)**  
Main execution metric. Lower is better.
"""
    )

with c2:
    st.markdown(
        """
**What is being optimized?**  
Execution quality under explicit liquidity constraints.

**What is not the question here?**  
This phase is **not** about predicting directional alpha.

So the objective is not:
- whether to take risk,
- nor in which direction to trade,
- but rather how to execute a known order efficiently.
"""
    )

callout(
    """
<b>Key distinction.</b><br>
In Phase 1, the side of the trade is already known.
The only question is:
<b>how should the order be executed?</b><br><br>
That is why the natural baselines are <b>TWAP</b> and <b>VWAP</b>,
not momentum or directional alpha rules.
""",
    tone="success",
)

section_title("Core question")

st.markdown(
    r"""
Suppose a trader wants to execute a parent order of size \(Q\).

Submitting the full quantity at once may:
- cross the spread immediately,
- consume visible liquidity,
- move deeper into the book,
- and therefore lead to poor execution quality.

The execution problem is therefore:

> can the parent order be split into child orders in a smarter way than simple standard schedules?
"""
)

section_title("Methods compared in this phase")

left, center, right = st.columns(3)

with left:
    st.subheader("Standard baselines")
    st.markdown(
        """
- **TWAP**  
  Split the order uniformly over time

- **VWAP**  
  Split the order according to an estimated volume profile
"""
    )

with center:
    st.subheader("Intermediate baselines")
    st.markdown(
        """
- **Liquidity-only**  
  Trade more where the local book looks easier to access

- **Alpha-only**  
  Trade more where short-term conditions suggest urgency
"""
    )

with right:
    st.subheader("Method: MALP")
    st.markdown(
        """
**MALP = Microstructure Alpha + Liquidity Policy**

MALP combines:
- local visible liquidity,
- spread and top-of-book depth,
- a short-term alpha score,
- and a smooth normalized schedule over child orders.
"""
    )

section_title("Why TWAP and VWAP matter here")

st.markdown(
    """
TWAP and VWAP are the classical execution baselines because they answer the same type of question as this phase:

- **TWAP** asks: what if we simply spread the order uniformly in time?
- **VWAP** asks: what if we concentrate the order where market volume is expected to be higher?

If a proposed method cannot beat these schedules, it is hard to argue that it brings execution value.
"""
)

callout(
    """
<b>Important.</b><br>
TWAP and VWAP appear in <b>Phase 1 only</b> because this is the only phase where the problem is
<b>execution of a known order</b>.<br><br>
In later phases, the problem changes to <b>directional alpha and robustness</b>,
so different baselines are needed.
""",
    tone="warn",
)

section_title("Execution model")

st.markdown(
    """
The benchmark uses a top-of-book execution model based on visible liquidity at levels 1 and 2.

The midprice is
"""
)

st.latex(r"m_t = \frac{p^{a,1}_t + p^{b,1}_t}{2}")

st.markdown("and the spread is")

st.latex(r"\mathrm{spr}_t = p^{a,1}_t - p^{b,1}_t")

st.markdown(
    r"""
For a buy child order of size \(q\), the simulated execution price depends on:
- how much of the ask queue is consumed at level 1,
- then at level 2,
- and whether the order needs to go beyond visible depth.
"""
)

st.latex(
    r"""
\pi_t^{\mathrm{BUY}}(q)
=
\frac{
q^{(1)} p_t^{a,1} + q^{(2)} p_t^{a,2} + q^{(3)}(p_t^{a,2}+\mathrm{spr}_t)
}{q}
"""
)

st.markdown(
    """
The sell-side version is defined symmetrically on the bid side.

This is a **top-of-book liquidity model**:
it captures visible spread and depth consumption, but it is not a full queue-priority simulator.
"""
)

section_title("Main metric: implementation shortfall")

st.markdown(
    """
The average execution price of the parent order is
"""
)

st.latex(
    r"""
\bar{\pi} = \frac{1}{Q}\sum_{t=1}^{T} q_t \pi_t(q_t)
"""
)

st.markdown(
    """
The implementation shortfall in basis points is
"""
)

st.latex(
    r"""
\mathrm{IS}_{\mathrm{bps}}
=
\mathrm{sign}(s)\,
\frac{\bar{\pi} - m_1}{m_1}\times 10^4
"""
)

st.markdown(
    """
Interpretation:
- for a **buy**, paying above the initial midprice is bad;
- for a **sell**, receiving below the initial midprice is bad;
- therefore **lower implementation shortfall is better**.
"""
)

callout(
    """
<b>Interpretation of bps here.</b><br>
A lower implementation shortfall means the execution policy obtained prices closer to the initial reference price.
So in this phase, unlike Phases 3 and 4, the key metric is a <b>cost / slippage metric</b>, not a profit metric.
""",
    tone="info",
)

section_title("What MALP adds")

st.markdown(
    """
MALP combines two types of information:

1. **Liquidity quality**  
   Where is the order book easier to access?

2. **Execution alpha / urgency**  
   Are short-term conditions suggesting that waiting may become more expensive?

The combined score is
"""
)

st.latex(
    r"""
z_t = \lambda_{\ell}\tilde \ell_t + \lambda_a \tilde a_t
"""
)

st.markdown(
    """
and this score is transformed into execution weights over time.

So MALP tries to answer both:
- **where** execution looks easier,
- and **when** delay may be costly.
"""
)

callout(
    """
<b>Interpretation.</b><br>
TWAP ignores local liquidity and urgency.<br>
VWAP uses expected volume but not local short-term microstructure conditions.<br>
MALP adds both a liquidity component and an urgency component.
""",
    tone="success",
)

section_title("What this phase proves")

st.markdown(
    """
This phase does **not** claim that the project already predicts directional returns.

Instead, it shows something more specific:

1. microstructure information is useful even before any directional trading problem is introduced;  
2. execution quality can be improved relative to simple schedules;  
3. combining liquidity and short-term execution alpha is more informative than using either one alone.
"""
)

section_title("Written summary")

summary_md = report_text_or_none("benchmark_summary.md")
if summary_md is None:
    st.info("benchmark_summary.md not found in reports/")
else:
    with st.expander("Open written phase-1 summary"):
        st.markdown(summary_md)

section_title("Visual summaries")

v1, v2 = st.columns(2)
with v1:
    show_image("real_mean_is_bps.png", "Phase 1 — real held-out mean implementation shortfall (bps)")
with v2:
    show_image("synthetic_mean_is_bps.png", "Phase 1 — synthetic mean implementation shortfall (bps)")

callout(
    """
<b>How to read the plots.</b><br>
These plots compare execution policies by their average implementation shortfall.<br>
Because lower is better, the best method is the one with the smallest bar or curve level.
""",
    tone="info",
)

section_title("Takeaway")

callout(
    """
<b>Main takeaway.</b><br>
Phase 1 shows that the project already extracts value from order-book information
before moving to any directional-alpha setting.<br><br>
It establishes a first result:
<b>microstructure-aware scheduling can improve execution relative to standard baselines such as TWAP and VWAP.</b>
""",
    tone="success",
)
