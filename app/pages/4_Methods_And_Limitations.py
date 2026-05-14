# app/pages/4_Methods_And_Limitations.py
import sys
from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.append(str(APP_DIR))

from common import badge, callout, doc_text_or_none, hero, inject_css

st.set_page_config(page_title="Methods & Limitations", layout="wide")
inject_css()

hero(
    "Methods & Limitations",
    "A guided explanation of the market concepts, mathematical quantities, cost model, drift handling, and modeling scope used throughout the project.",
)

st.markdown(
    badge("microstructure", "method") + badge("scope", "method") + badge("limitations", "method"),
    unsafe_allow_html=True,
)

callout(
    """
<b>What this page is for.</b><br>
This page explains the main concepts used across the project:
what the order book contains, what a parent order is, what a basis point means,
how costs are modeled, why different phases use different baselines,
and what is deliberately left outside the simulator.
""",
    tone="info",
)

st.subheader("1. Basic market concepts")

c1, c2 = st.columns(2)

with c1:
    st.markdown(
        """
**Bid**  
Best currently visible buy price.

**Ask**  
Best currently visible sell price.

**Parent order**  
A larger order that must be executed over time.

**Child orders**  
Smaller pieces of the parent order sent sequentially.
"""
    )

with c2:
    st.markdown(
        """
**Directional alpha**  
A signal that tries to predict short-horizon price moves.

**Execution**  
The problem of how to trade a known order efficiently.

**Basis point (bp)**  
A unit of return or cost:
- 1 bp = 0.01%
- 10 bps = 0.10%
- 100 bps = 1%
"""
    )

callout(
    """
<b>Key distinction.</b><br>
The project separates two different problems:
<b>execution</b> and <b>directional alpha</b>.<br><br>
Execution asks:
<i>given a known order, how should it be traded?</i><br>
Alpha asks:
<i>should the strategy take risk, in what direction, and with how much model trust?</i>
""",
    tone="success",
)

st.subheader("2. Observed market state")

st.markdown(
    """
The project uses a top-of-book representation of the market.
At each step, it observes prices and quantities at the best levels of the book.
"""
)

st.latex(r"m_t = \frac{p^{a,1}_t + p^{b,1}_t}{2}")
st.markdown("**Midprice**: neutral reference price between the best bid and ask.")

st.latex(r"\mathrm{spr}_t = p^{a,1}_t - p^{b,1}_t")
st.markdown("**Spread**: immediate crossing cost and a key liquidity signal.")

st.latex(
    r"""
\mathrm{imb}^{(1)}_t
=
\frac{q^{b,1}_t - q^{a,1}_t}{q^{b,1}_t + q^{a,1}_t + \varepsilon}
"""
)
st.markdown(
    "**Level-1 imbalance**: summary of local asymmetry between visible bid-side and ask-side liquidity."
)

st.latex(
    r"""
\mu_t
=
\frac{p^{a,1}_t q^{b,1}_t + p^{b,1}_t q^{a,1}_t}{q^{b,1}_t + q^{a,1}_t + \varepsilon}
"""
)
st.markdown(
    "**Microprice**: spread-aware local price proxy that reacts to book imbalance and local pressure."
)

callout(
    """
<b>Interpretation.</b><br>
If the spread widens, trading becomes more expensive.<br>
If imbalance becomes strongly positive or negative, the local pressure in the book becomes informative.<br>
If the microprice deviates from the midprice, the order book may contain short-horizon directional information.
""",
    tone="info",
)

st.subheader("3. Why Phase 1 is different from Phases 3 and 4")

left, right = st.columns(2)

with left:
    st.markdown(
        """
### Execution layer

In Phase 1, the problem is:

> given a parent order, how should child orders be scheduled?

That is why the relevant baselines are:
- TWAP
- VWAP
- Liquidity-only
- Alpha-only
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

    st.markdown("**Implementation shortfall** is the main execution metric. Lower is better.")

with right:
    st.markdown(
        """
### Alpha layer

In Phases 3 and 4, the problem changes:

> should the strategy take directional exposure, in what direction, and under what degree of model trust?

That is why the baselines change to:
- momentum
- signed_flow
- micro_fixed
- ridge_static
- online_regime
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
Here:
- \(g_t\) is gross P\&L,
- \(c_t\) is transaction cost,
- \(n_t\) is net P\&L.
"""
    )

callout(
    """
<b>Why TWAP and VWAP do not appear later.</b><br>
TWAP and VWAP are execution schedules. They are appropriate only when the side and the parent order are already known.
In the later phases, the project studies <b>directional alpha</b> and <b>robustness under shift</b>,
so different baselines are needed.
""",
    tone="warn",
)

st.subheader("4. Cost model and economically meaningful metrics")

m1, m2 = st.columns(2)

with m1:
    st.markdown(
        """
The project focuses on economically meaningful quantities:
- mean net bps
- total net bps
- turnover
- hit rate
- positive episodes
- episode-level t-stat
"""
    )

with m2:
    st.markdown(
        """
These metrics are preferred to raw prediction accuracy because a strategy may:
- predict often but trade too much,
- look good before costs but fail after costs,
- or perform well on average but be unstable across episodes.
"""
    )

callout(
    """
<b>Main idea.</b><br>
The project is designed to answer:
<i>does the method remain useful after trading costs?</i><br>
That is why spread-based cost penalties and turnover matter so much in the benchmark.
""",
    tone="success",
)

st.subheader("5. Drift handling")

st.markdown(
    """
The project includes two layers of adaptation:

1. **Warmup adaptation within an episode**  
   The strategy uses the beginning of the current episode to infer whether the local regime looks aligned or inverted.

2. **Domain-shift fallback across assets or venues**  
   The strategy uses a shift score to decide whether the stronger learned model should be trusted.
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

callout(
    """
This makes model trust <b>conditional on shift</b>, rather than fixed by default.
The strategy does not assume that the same learned model should always be trusted equally across all markets.
""",
    tone="success",
)

st.subheader("6. What is modeled explicitly")

st.markdown(
    """
- spread and displayed top-of-book liquidity  
- level-1 / level-2 imbalance  
- signed trade flow  
- explicit spread-based transaction costs  
- turnover penalties  
- walk-forward retraining  
- warmup adaptation  
- cross-venue fallback under shift  
"""
)

st.subheader("7. What is not modeled explicitly")

st.markdown(
    """
- exact queue position and passive fill probability  
- time-priority mechanics inside the limit-order queue  
- detailed cancellation / replacement dynamics  
- latency-sensitive exchange behavior  
- full production-grade order management  
"""
)

callout(
    """
The correct interpretation is therefore:
<b>research-grade microstructure benchmark</b>, not a full exchange simulator.
""",
    tone="warn",
)

st.subheader("8. Why queue position is not modeled here")

st.markdown(
    """
A full queue-position model would require a much more detailed description of passive order dynamics,
including exact time priority, cancellations, and partial fills.

The current benchmark instead focuses on:
- visible top-of-book liquidity,
- explicit spread-based costs,
- and robust comparative evaluation across strategies.

A queue-aware passive fill model could be added later,
but it would need to be presented honestly as an approximation under market-by-price data,
not as an exact queue simulator.
"""
)

st.subheader("9. Why the benchmark structure is coherent")

st.markdown(
    """
The benchmark follows a progression:

- **Phase 1:** can order-book information improve execution relative to standard schedules?  
- **Phase 2:** can more advanced alpha ideas be tested in a controlled but realistic sandbox?  
- **Phase 3:** do those ideas survive on real historical data after costs?  
- **Phase 4:** do they remain useful when the market changes?

This separation makes the role of each baseline and each method much easier to interpret.
"""
)

callout(
    """
<b>Project logic.</b><br>
The dashboard should be read as:
<b>execute better → test safely → validate on real data → control model risk under shift.</b>
""",
    tone="info",
)

st.subheader("10. Supporting method notes")

for filename, title in [
    ("method_note.md", "General method note"),
    ("alpha_research_method.md", "Phase 2 method note"),
    ("alpha_phase3_method.md", "Phase 3 method note"),
    ("alpha_phase4_method.md", "Phase 4 method note"),
]:
    text = doc_text_or_none(filename)
    with st.expander(title):
        if text is None:
            st.info(f"{filename} not found in docs/")
        else:
            st.markdown(text)
