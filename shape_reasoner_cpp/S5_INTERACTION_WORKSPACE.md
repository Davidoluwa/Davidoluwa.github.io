# SERA S5: Shift-Equivariant 2-D Interaction Workspace

## Status

Research prototype. This document records one derivation, one falsified
hypothesis, one experimental-design failure, and the architecture that
survived them. It is not an ASI claim and does not establish superiority over
a transformer, which has not been run as a matched baseline.

## 1. The CP relation is a tensor factorization

S4 exposes operand interaction only through

$$q_{pr}=P_{pr}\ell_r\rho_r,\qquad
\ell_r=\sum_i L_{ir}a_i,\quad \rho_r=\sum_j R_{jr}b_j .$$

Substituting,

$$q_{pr}=P_{pr}\sum_{i,j}L_{ir}R_{jr}\,a_ib_j
\;\Longrightarrow\;
q_p=\sum_{i,j}T_{pij}\,a_ib_j,\qquad
T_{pij}=\sum_{r=1}^{R}P_{pr}L_{ir}R_{jr}.$$

So the relational feature is exactly a contraction of the operand pair with a
tensor $T$ **constrained to CP rank at most $R$** (`shared_rank`).

Unsigned multiplication needs the partial-product sums
$s_p=\sum_{i+j=p}a_ib_j$, i.e. the truncated polynomial-multiplication tensor

$$T^{\times}_{pij}=\mathbf 1[i+j=p].$$

Its CP rank is a classical bilinear-complexity quantity: multiplying two
$W$-term polynomials requires $2W-1$ multiplications over an infinite field
(Winograd), achieved by evaluation–interpolation at $2W-1$ points.

`research/cp_rank_bound.py` measures this directly by alternating least
squares with random restarts. Relative Frobenius error of the best rank-$R$
fit:

| $R$ | $W=6$ | $W=8$ | $W=12$ |
|---:|---:|---:|---:|
| $W$ | 1.0e-2 | 1.3e-2 | 2.0e-2 |
| $2W-1$ | **1.3e-7** | 2.0e-3 | — |
| $2W$ | 1.4e-8 | **4.9e-7** | — |

Error collapses by five orders of magnitude at $R\approx 2W-1$ and is
strictly nonzero below it. The shipped S4 configuration ($W=12$, $R=12$)
sits at 2.0e-2: **its CP relation cannot represent multiplication.** That
much is solid.

## 2. The hypothesis that failed

The tempting conclusion — *therefore S4 cannot multiply* — is **false**, and
the experiment falsified it.

The CP relation is not the model's only route to operand interaction. S4 also
runs 20 recurrent steps of a gated field over bit positions, with the raw
bits $a_p,b_p$ present as features at every position. That recurrence is a
1-D convolutional-recurrent machine in its own right and can implement
shift-and-add algorithmically, never routing multiplication through $q$ at
all.

A rank bound on one pathway is not a bound on the model. The correct
statement is narrower:

> Widening the CP relation until it can represent multiplication costs
> $3W(2W-1)=\Theta(W^2)$ parameters in that relation alone. Whether the model
> needs the relation for multiplication is a separate, empirical question.

## 3. The experiment-design failure

The first sweep was run at $W=6$ so that evaluation could be exhaustive over
every admissible operand pair. That choice destroyed the phenomenon under
study. Under the shared validity convention a case is admissible only if the
true result fits in $W$ bits, so at $W=6$ every product is at most 63 and
multiplication collapses to a small table.

Measured result: **every** configuration solved it — S4 at rank 4 (1,865
parameters) reached 100% exact multiplication accuracy, as did ranks 8, 11,
16, and S5. Seed-to-seed variation was the only signal.

$W=6$ cannot discriminate architectures. The informative regime is $W=12$,
where S4 was reported at 26.6% in-domain multiplication. Exhaustive
evaluation is not affordable there, so the harness scores a fixed sampled set
of admissible cases, identical across every architecture and seed.

## 4. Architecture

S5 keeps S4's commitments — no attention, no token autoregression, one
parameter set tied across positions, reasoning steps, operators, and proof
nodes, operator supplied only as literal bits — and changes where interaction
lives.

**Workspace.** Lattice cell $(i,j)$ observes

$$\phi_{ij}=[\,a_i,\;b_j,\;a_ib_j,\;\operatorname{bits}(o),\;
\mathbf 1_{i=0},\mathbf 1_{j=0},\mathbf 1_{i=W-1},\mathbf 1_{j=W-1}\,],$$

$$g^0_{ij}=\tanh(V\phi_{ij}+b_V).$$

The term $a_ib_j$ is the complete second-order interaction of the two operand
fields — the same object $T$ was approximating, supplied exactly and at zero
parameter cost.

**Transitions.** With a shared projection $z_{ij}=Bg_{ij}$ and neighbourhood
$\mathcal N$ = the nine $3\times3$ offsets plus an anchor tap onto $z^0$:

$$c^{t+1}_{ij}=\tanh\Big(\sum_{d\in\mathcal N}A_d\,z^t_{i+d_1,j+d_2}+b_c\Big),
\qquad
u^{t+1}_{ij}=\sigma\Big(\sum_{d\in\mathcal N}U_d\,z^t_{i+d_1,j+d_2}+b_u\Big),$$

$$g^{t+1}_{ij}=(1-u^{t+1}_{ij})\odot g^t_{ij}+u^{t+1}_{ij}\odot c^{t+1}_{ij}.$$

$A_d,U_d$ depend only on the **offset**, never on $(i,j)$: the stencil is
translation-equivariant and tied across all cells and all steps. Cells
outside the lattice read as zero.

**Readout.** Answer bit $p$ is read from cell $(p,0)$:
$\Pr(y_p=1)=\sigma(w_o^\top g^T_{p,0}+b_o)$.

Nothing arithmetic is written into the model. Translation equivariance is the
generic lattice prior a convolution has; no carry rule, shift amount,
anti-diagonal sum, or operation identity appears in the code, and which cell
holds which output bit is a fixed readout convention the stencil must learn to
route into.

## 5. Parameter accounting

For $F=15$ features, $C$ channels, shared rank $S$, and $K=10$ taps:

$$P_{\text{S5}}=CF+C+SC+2KCS+2C+C+1 .$$

At $C=12,S=5$: **1,489 parameters**, and

$$\frac{\partial P_{\text{S5}}}{\partial W}=0 .$$

The parameter count is independent of bit width — verified as a unit test
(`check_width_independent_parameters`). Only the compute horizon scales, as
$2W$ steps over $W^2$ cells. By contrast the S4 CP relation alone costs
$3WR$, and $3W(2W-1)$ if widened to representability: 828 parameters at
$W=12$, growing quadratically.

This is a statement about parameter growth, not about accuracy. Section 6 of
`S5_RESULTS.md` reports what the two architectures actually achieve.

## 6. Correctness of the implementation

The backward pass is hand-written. It is checked against central finite
differences per tensor, with an epsilon sweep, because parameters are stored
in float32 and a step of $10^{-4}$ is dominated by round-off cancellation
rather than truncation error:

| $\epsilon$ | worst relative error over all tensors |
|---:|---:|
| 1e-3 | 8.0e-3 |
| 3e-3 | 3.9e-3 |
| 1e-2 | 1.1e-3 |

The suite additionally covers inference determinism, loss reduction,
bit-width rejection, invalid configuration rejection, bit-exact checkpoint
round trip, shape-mismatch rejection, evaluation-time parameter immutability,
and width-independence of the parameter count.

## 7. Step-by-step reasoning

`sera_s5_proof` applies the single frozen kernel at every internal node of an
expression tree in post-order, consuming its own predicted child values.
The derivation is the evaluation order; no text is generated and no search is
performed. Two step metrics are reported and deliberately not merged:

- **semantic** — the step equals the true value of that subtree;
- **local** — the step is a consistent transformation of the child values it
  was actually handed, which may themselves be wrong.

High local validity with low semantic accuracy is the signature of a correct
kernel fed corrupted inputs, and collapsing the two would overstate the
result.
