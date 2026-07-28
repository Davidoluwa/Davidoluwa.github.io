# SERA S3 Recurrent Neural Shape Field

## Purpose

SERA S3 replaces S2's fresh symbolic program search with a small reusable
neural process. It remains non-transformer: there is no attention, softmax,
tokenization, positional encoding, or autoregressive reasoning.

The architecture was informed by the permutation-invariant aggregation
principle in Deep Sets and by the reusable function-prior objective of
Conditional Neural Processes:

- https://arxiv.org/abs/1703.06114
- https://arxiv.org/abs/1807.01613

SERA differs from both by recurrently constructing a complete neural field over
a discrete state graph and freezing that field before prediction.

## Two levels of state

### Reusable global parameters

\[
\theta =
\{W_{\mathrm{in}},W_s,W_n,W_a,w_o,b_{\mathrm{in}},b_r,b_o\}.
\]

The same 1,993 scalar parameters are used for every state, output wire,
reasoning step, episode, and relation.

### Episode-specific reasoning shape

\[
\mathcal S_E =
\{h_{j,v}^{(T)}:
j\in\{1,\ldots,6\},
v\in\{0,1\}^6\}.
\]

This \(6\times64\times24\) activation field is constructed once from an
episode's support set. It is not a sequence of emitted symbols. After
construction it is frozen, decoded into a 64-state output field, and queried
by lookup.

## Input initialization

For output wire \(j\) and Boolean state \(v\), the node receives only:

\[
x_{j,v} =
[m_v,\;m_v\widetilde y_{j,v},\;v_1,\ldots,v_6],
\]

where \(m_v\) is the observation mask and \(\widetilde y\in\{-1,+1\}\).
Missing labels are represented by zero. No Hamming-weight or arithmetic
feature is supplied.

\[
h_{j,v}^{(0)}
=
\tanh(W_{\mathrm{in}}x_{j,v}+b_{\mathrm{in}}).
\]

## Recurrent shape construction

Each Boolean state is adjacent to the six states differing in exactly one raw
input wire. At reasoning step \(t\):

\[
\bar h_{j,v}^{(t)}
=
\frac{1}{6}
\sum_{u:d_H(u,v)=1}h_{j,u}^{(t)}.
\]

\[
c_{j,v}^{(t+1)}
=
\tanh(
W_s h_{j,v}^{(t)}
+W_n\bar h_{j,v}^{(t)}
+W_a h_{j,v}^{(0)}
+b_r
).
\]

\[
h_{j,v}^{(t+1)}
=
(1-\alpha)h_{j,v}^{(t)}
+\alpha c_{j,v}^{(t+1)},
\qquad \alpha=0.5.
\]

Six iterations are executed. The anchoring term preserves raw evidence while
the residual path stabilizes recurrent optimization.

## Readout

\[
\ell_{j,v}=w_o^\top h_{j,v}^{(T)}+b_o,
\qquad
\hat y_{j,v}=\mathbf 1[\ell_{j,v}\ge0].
\]

Observed support labels are copied exactly into the frozen output field.
Unobserved states use the learned readout. This makes zero support
reproduction error an integrity property, not a learning score.

## Meta-training

The support set is a corrupted partial view of an anonymous complete relation.
The model minimizes weighted binary cross-entropy:

\[
\mathcal L(\theta)
=
\frac{
\sum_{j,v}w_v\,
\operatorname{BCE}(\ell_{j,v},y_{j,v})
}{
\sum_{j,v}w_v
},
\]

with \(w_v=1\) for hidden states and \(w_v=0.1\) for observed states.

Adam updates \(\theta\) across 8,000 independently scrambled training
episodes. Support sizes are sampled from
\(\{12,16,24,32,40,48,56,60\}\). No operation label enters the learner.

## Parameter count

With \(n=6\) raw wires and hidden width \(H=24\):

\[
\begin{aligned}
P
&=H(n+2)+H
  +3H^2+H
  +H+1\\
&=24(8)+24+3(24^2)+24+24+1\\
&=1,993.
\end{aligned}
\]

The weights occupy roughly 7.8 KiB as 32-bit floats.

## Complexity

For \(m\) output wires, \(V=2^n\) state nodes, hidden width \(H\), and \(T\)
reasoning steps:

\[
\text{compile time}
=
O\!\left(mTV(H^2+nH)\right).
\]

\[
\text{shape memory}=O(mVH).
\]

The parameter count is only:

\[
O(H^2+nH),
\]

and is independent of the number of episodes. However, enumerating every
Boolean state leaves activation memory and compilation exponential in \(n\).

The next architecture must preserve shared recurrent parameters while
constructing a sparse graph containing only observed, queried, and
uncertainty-relevant states.
