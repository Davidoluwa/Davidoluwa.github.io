# SERA S1 Mathematics

## 1. Research object

S1 represents a problem as a typed relational shape rather than a sequence of
tokens. A shape is a pairwise factor graph

\[
\mathcal G=(V,F),\qquad x_i\in\{0,\ldots,K_i-1\}.
\]

Each node has a unary cost \(u_i(x_i)\). Each factor
\(f=(i,j)\) has a general, finite cost table \(\psi_f(x_i,x_j)\). The global
shape energy is

\[
E(x)=\sum_{i\in V}u_i(x_i)
    +\sum_{f=(i,j)\in F}\psi_f(x_i,x_j).
\]

The reasoning problem is the MAP problem

\[
x^\star=\operatorname*{arg\,min}_{x}E(x).
\]

This is the system's mathematical meaning of “forming the shape up front.”
Language is not part of this optimization. A later compiler receives the
settled assignment, confidences, supporting factors, and contradiction
telemetry.

## 2. Counterfactual shape messages

Let \(m_{f\to i}^{(t)}(a)\) be the message arriving at node \(i\) from factor
\(f\), evaluated at candidate state \(a\). The full local belief-cost surface
is

\[
B_i^{(t)}(a)
  =u_i(a)+\sum_{g\in N(i)}m_{g\to i}^{(t)}(a).
\]

For \(f=(i,j)\), remove the message that came through the same factor:

\[
C_{i\setminus f}^{(t)}(a)
  =B_i^{(t)}(a)-m_{f\to i}^{(t)}(a).
\]

The proposed message to \(j\) is a min-plus tensor contraction:

\[
\widehat m_{f\to j}^{(t+1)}(b)
  =\min_a\left[
      \psi_f(a,b)+C_{i\setminus f}^{(t)}(a)
    \right].
\]

For every possible destination state \(b\), this asks a counterfactual
question: “What is the cheapest complete upstream stance compatible with
\(x_j=b\)?” One update therefore moves a relational cost surface, not a word
or a token.

Messages are gauge-normalized because adding the same scalar to every state
does not alter an argmin:

\[
\bar m(b)=\widehat m(b)-\min_{b'}\widehat m(b').
\]

Damping prevents discontinuous oscillation:

\[
\widetilde m^{(t+1)}
  =m^{(t)}+\lambda\left(\bar m-m^{(t)}\right),
\qquad 0<\lambda\le 1.
\]

S1 uses \(\lambda=0.72\) in the stress suite.

## 3. Tension and deterministic active-frontier settlement

The factor-to-node tension is the infinity-norm residual

\[
r_d^{(t)}
  =\left\|\widetilde m_d^{(t+1)}-m_d^{(t)}\right\|_\infty ,
\]

where \(d\) identifies a directed factor message. A proposal is committed
only when

\[
r_d^{(t)}>\varepsilon.
\]

All active proposals in a sweep are computed from the same old message
field. They are committed together. This preserves deterministic synchronous
semantics even when factor storage order is shuffled.

Let \(\operatorname{down}(d)\) be the messages whose source node receives
message \(d\), excluding the immediate reverse message through the same
factor. The next active frontier is

\[
A_{t+1}
  =\bigcup_{\substack{d\in A_t\\r_d^{(t)}>\varepsilon}}
    \left(\{d\}\cup\operatorname{down}(d)\right).
\]

The message remains active while damping is still changing it, and only
causally downstream messages are reawakened. Settlement terminates when
\(A_{t+1}=\varnothing\), or reports non-convergence at the sweep limit.

For homogeneous state count \(K\), dense work is

\[
C_{\text{dense}}=2|F|K^2S,
\]

for \(S\) sweeps. Active-frontier counterfactual work is

\[
C_{\text{active}}=K^2\sum_{t=0}^{S-1}|A_t|.
\]

The observed candidate reduction is

\[
R_{\text{active}}
  =1-\frac{C_{\text{active}}}{C_{\text{dense}}}.
\]

This measures only the expensive \(K^2\) counterfactual contractions. Belief
reconstruction, clocks, hashing, output, and standard-library overhead are
not hidden inside this number.

## 4. Readout without language

After settlement,

\[
\hat x_i=\operatorname*{arg\,min}_a B_i(a).
\]

For uncertainty telemetry only, S1 turns belief costs into a tempered
distribution:

\[
q_i(a)=
\frac{\exp\left(-(B_i(a)-\min_cB_i(c))/T\right)}
     {\sum_b\exp\left(-(B_i(b)-\min_cB_i(c))/T\right)}.
\]

Confidence and entropy are

\[
\operatorname{conf}_i=\max_a q_i(a),\qquad
H_i=-\sum_a q_i(a)\log q_i(a).
\]

These quantities do not feed token generation. They decide whether a
correlated alternative should be preserved as a shape particle.

## 5. Adaptive correlated shape particles

Independent marginal argmins can destroy a globally correlated alternative.
S1 branches when settlement fails or when the most uncertain unclamped node
exceeds the entropy threshold:

\[
i^\star=\operatorname*{arg\,max}_{i\notin C}H_i.
\]

For candidate state \(s\), a child particle receives the clamp

\[
u_{i^\star}^{(s)}(a)
  =u_{i^\star}(a)+M\,\mathbf 1[a\ne s],
\]

with large finite \(M\). Each child settles as a complete shape. A child is
split again if it remains unsettled or uncertain, subject to depth and beam
budgets.

Particles are ranked using original, unclamped energy:

\[
\mathcal F_p
  = E_{\text{original}}(\hat x_p)
    +\tau\sum_iH_i^{(p)}
    +\beta C_p,
\]

where \(C_p\) is its counterfactual candidate count. Assignment hashes remove
duplicate modes.

This is not a claim that particles themselves are novel. The specific SERA
candidate is the closed coupling of:

1. counterfactual min-plus shape messages;
2. synchronous causal active-frontier settlement;
3. telemetry-triggered recursive correlated branching;
4. goal-subshape extraction; and
5. a separate, cheap semantic compiler.

## 6. Contradictions and goal shapes

The solver always evaluates the proposed assignment under the original
energy \(E(\hat x)\). For constraint-like factors, S1 reports a factor
violation when

\[
\psi_f(\hat x_i,\hat x_j)
  >\min_{a,b}\psi_f(a,b)+\delta.
\]

That count is meaningful for the modular constraint tests. For arbitrary
soft cost tables it is only a diagnostic; optimal solutions can legitimately
select a non-minimal local factor to reduce global energy.

Goal-shape extraction takes the connected component containing the query
node. Disconnected distractors are excluded before settlement. This is an
exact topological operation, not an attention heuristic.

## 7. What is standard, and what remains research

Factor graphs and local message passing are established techniques; the
canonical factor-graph treatment is
[Kschischang, Frey, and Loeliger](https://ieeexplore.ieee.org/document/910572/).
Residual scheduling is also established, including asynchronous residual
belief propagation by
[Elidan, McGraw, and Koller](https://arxiv.org/abs/1206.6837).
Particle belief propagation has a substantial prior literature; examples
include [Ihler and McAllester](https://proceedings.mlr.press/v5/ihler09a.html)
and differentiable nonparametric BP by
[Opipari et al.](https://arxiv.org/abs/2303.04616). Convergent or bounded
loopy alternatives such as
[tree-reweighted message passing](https://proceedings.mlr.press/r5/kolmogorov05a/kolmogorov05a.pdf)
are important comparators for a future stage.

S1 therefore makes no novelty claim for min-sum, residuals, factor graphs, or
particles in isolation. The empirical question is whether their SERA-specific
composition is a useful substrate for non-linguistic shape formation.

## 8. Guarantees and limits

- On trees, min-sum messages recover the exact MAP solution under ordinary
  uniqueness conditions; S1 checks this against exhaustive enumeration.
- On loopy graphs, plain settlement is approximate and may not converge.
- Adaptive conditioning can break difficult correlations but is exponential
  in the worst case and is controlled by explicit budgets.
- The current engine assumes finite pairwise categorical factors.
- Learning factors, learning topology, continuous latent shapes, temporal
  memory, and grounding are not solved in S1.

