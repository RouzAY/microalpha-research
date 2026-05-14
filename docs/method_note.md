# Method note

## Goal

We want to execute a parent order over a fixed horizon while minimizing implementation shortfall.

For a buy order with average execution price \(\bar p\) and arrival mid-price \(m_0\), we report

\[
\text{IS}_{\text{bps}} = 10^4\, \frac{\bar p - m_0}{m_0}.
\]

For a sell order we flip the sign so that **lower is always better**.

## Side-specific execution alpha target

For each book update \(t\), define a small child-order execution cost now and compare it with the average cost over a short future horizon.

For buys, the model target is

\[
y_t^{(buy)} = \frac{\mathbb{E}[c^{(buy)}_{t+1:t+H}] - c_t^{(buy)}}{m_t},
\]

so it is positive when executing now is better than waiting.

For sells,

\[
y_t^{(sell)} = \frac{c_t^{(sell)} - \mathbb{E}[c^{(sell)}_{t+1:t+H}]}{m_t}.
\]

We fit a ridge regression on handcrafted book/trade features.

## Liquidity score

For buys, we use opposite-side liquidity:

\[
L_t^{(buy)} = \log(\text{ask depth}_{1+2}) - \log(\text{spread}_t + \varepsilon).
\]

For sells, the same on the bid side.

## MALP score and schedule

The final score is

\[
S_t = z(L_t) + \lambda z(A_t),
\]

where \(A_t\) is the model prediction and \(z(\cdot)\) is standardization inside the execution window.

The normalized schedule weights are

\[
w_t = \frac{\exp(\text{clip}(S_t,-1.5,1.5))}{\sum_u \exp(\text{clip}(S_u,-1.5,1.5))}.
\]

The parent order quantity \(Q\) is allocated as cumulative targets

\[
q_t = \max\{0, Q\sum_{u \le t} w_u - \sum_{u < t} q_u\}.
\]

This keeps the policy deterministic, reproducible, and interview-friendly.
