# SERA S4: CP-Relational Gated Neural Proof Kernel

## Research claim

S4 is a C++20, non-transformer neural prototype for **shape-before-language**
reasoning. It does not reason by generating tokens. An external parser turns a
mathematical expression into a syntax tree, one shared neural kernel evaluates
every internal node, and a deterministic renderer translates the resulting
proof DAG into high-level steps.

This is not an ASI claim. The current checkpoint fails the strict research
gate, principally on multiplication and recursive error accumulation.

## Information boundary

For operands \(a,b\), bit position \(p\), and raw operator byte \(o\), the
learner receives only:

\[
x_p=[a_p,b_p,\operatorname{bits}(o),\mathbf 1_{p=0},
\mathbf 1_{p=W-1}].
\]

The eight operator features are the literal bits of `+`, `-`, or `*`, not a
privileged operation ID. Training provides the result bit vector for a whole
node. The learner never receives carries, borrows, partial products,
intermediate algorithms, evaluator corrections, or test labels.

Expression-tree structure is treated as problem syntax, not hidden guidance.
At inference the same frozen kernel consumes the predicted child values at
every parent node.

## Relational shape

A low-rank CP contraction exposes generic pairwise relationships between the
two operand fields:

\[
\ell_r=\sum_i L_{ir}a_i,\qquad
\rho_r=\sum_j R_{jr}b_j,
\]

\[
q_{pr}=P_{pr}\ell_r\rho_r.
\]

The factors \(L,R,P\) are learned. No diagonal, carry, shift, or
operation-specific relation is fixed in code. The augmented field is
\(e_p=[x_p,q_p]\).

This is the main “shape” mechanism: the model constructs a compact relational
object over both operands before producing any answer representation.

## Reusable gated field

Initial state:

\[
h_p^0=\tanh(W_{\mathrm{in}}e_p+b_{\mathrm{in}}).
\]

A single projection \(B\in\mathbb R^{R\times H}\) is shared by both recurrent
paths:

\[
z_p^t=Bh_p^t.
\]

For neighborhood \(\mathcal N=\{p,p-1,p+1,p^0\}\), the candidate and update
gate are

\[
c_p^{t+1}=\tanh\left(
\sum_{d\in\mathcal N}A_dz_d^t+b_c
\right),
\]

\[
g_p^{t+1}=\sigma\left(
\sum_{d\in\mathcal N}U_dz_d^t+b_g
\right),
\]

\[
h_p^{t+1}=(1-g_p^{t+1})\odot h_p^t+
g_p^{t+1}\odot c_p^{t+1}.
\]

The anchor term \(p^0\) re-injects the initial state projection. All matrices
are tied across bit positions, 20 recurrent steps, raw operators, arithmetic
examples, and proof-tree nodes. There is no attention matrix, positional token
sequence, or decoder loop.

Result bits are read in parallel:

\[
\Pr(y_p=1)=\sigma(w_o^\top h_p^T+b_o).
\]

## Temporal learning objective

The latter half of recurrent time is supervised using the same final result,
with linearly increasing temporal weights:

\[
\mathcal L =
\sum_{t=\lfloor T/2\rfloor}^{T}
\frac{t-\lfloor T/2\rfloor+1}
{\sum_{k=1}^{T-\lfloor T/2\rfloor+1}k}
\operatorname{BCE}(y,\hat y^t)
\lambda\sum_r\|\Theta_r\|_2.
\]

This supplies no intermediate arithmetic answer. It makes convergence
observable and shortens the gradient path. The group penalty acts on every
shared rank channel across the CP factors, projection, candidate lifts, and
gate lifts.

## Parameter accounting

For base feature count 12, bit width \(W\), hidden width \(H\), and shared rank
\(R\):

\[
P_{\text{S4}}=16H+10HR+3WR+1.
\]

The tested \(W=12,H=32,R=12\) model has:

\[
P_{\text{S4}}=4{,}785.
\]

The reported dense analogue replaces eight factorized recurrent maps with
dense \(H\times H\) maps and the CP relation with a dense \(W^3\) tensor:

\[
P_{\text{dense}}=H(12+R)+4H+8H^2+W^3+1=10{,}817.
\]

The structural reduction is \(55.76\%\). Parameter tying across 20 time steps
and all proof nodes adds no parameters. These counts establish compactness,
not superior sample efficiency to transformers; no matched transformer
baseline has yet been run.

## Complexity

Ignoring small feature terms, one neural node costs

\[
O\!\left(TW(HR+H)+WR\right)
\]

time and \(O(TWH)\) training memory. Inference can be reduced to \(O(WH)\)
state memory by removing caches. A proof tree with \(N\) internal nodes costs
\(N\) kernel evaluations but still uses one parameter set.

## Research lineage

The design combines, but does not duplicate:

- recurrent convolutional algorithm learning from the
  [Neural GPU](https://arxiv.org/abs/1511.08228);
- explicit relational inductive bias from
  [Graph Networks](https://arxiv.org/abs/1806.01261);
- tensor-factorized parameter efficiency from
  [Tensorizing Neural Networks](https://papers.neurips.cc/paper/5787-tensorizing-neural-networks.pdf)
  and [TT-RNN](https://arxiv.org/abs/1705.08052);
- reusable neural computation across program steps from
  [Neural Programmer-Interpreters](https://arxiv.org/abs/1511.06279).

S4's particular CP operand relation, shared projection for candidate and gate,
and proof-DAG evaluation are experimental engineering choices. Novelty has not
been established by an exhaustive prior-art or patent search.
