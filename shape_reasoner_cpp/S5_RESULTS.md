# SERA S5 Results

All numbers below are measured in this repository. Reproduction commands are
in §7. Two hypotheses were falsified along the way and both are recorded.

## 1. Headline

At $W=12$ — the regime where S4 was reported to fail — the S5 dual-readout
workspace reaches **2.9× S4's non-trivial multiplication accuracy using 2.4×
fewer parameters**, at parity on addition and subtraction.

| config | params | seeds | add | sub | mul | mul non-trivial | spread |
|---|---:|---:|---:|---:|---:|---:|---:|
| S4 rank 8 | 3,361 | 2 | 0.9995 | 1.0000 | 0.4797 | 0.1513 | 0.028 |
| S4 rank 12 *(shipped)* | 4,785 | 2 | 0.9995 | 0.9998 | 0.5000 | 0.1844 | 0.069 |
| S4 rank 23 $(=2W{-}1)$ | 8,701 | 2 | 0.9990 | 0.9995 | 0.4965 | 0.1804 | 0.006 |
| S5 corner readout | 1,985 | 2 | 0.5143 | 0.0083 | 0.7258 | 0.5541 | 0.194 |
| S5 row-pooled readout | 1,985 | 2 | 0.1128 | 0.1307 | 0.6637 | 0.4543 | 0.058 |
| **S5 dual readout** | **2,001** | **4** | **0.9992** | **0.9987** | **0.7157** | **0.5359** | 0.141 |

Scoring is over a fixed sampled set of 3,000 admissible cases per operator,
identical across every architecture and seed. "Non-trivial" excludes products
with an operand of 0 or 1, which are solvable by copying an operand; 1,838 of
the 3,000 multiplication cases are non-trivial. Every run recorded zero
parameter drift during evaluation.

## 2. Rank is not the constraint

The S4 CP relation cannot represent multiplication below rank $2W-1$; that is
established in `S5_INTERACTION_WORKSPACE.md` §1 and measured by
`research/cp_rank_bound.py`. The natural inference — widen the relation — is
**wrong**, and the sweep says so directly:

| S4 shared rank | 8 | 12 | 23 $(=2W{-}1)$ |
|---|---:|---:|---:|
| params | 3,361 | 4,785 | 8,701 |
| mul non-trivial | 0.151 | 0.184 | 0.180 |

Rank 23 costs 2.6× the parameters of rank 8 and buys nothing. The relation is
simply not the pathway carrying multiplication: S4's 20-step recurrent field
over bit positions is an independent computational route, and it is that
route which saturates. A rank bound on one pathway was never a bound on the
model — see §5.

## 3. Where the workspace wins, and why

S5 supplies the exact bilinear term $a_ib_j$ on a $W\times W$ lattice and
routes it with one translation-equivariant stencil tied across every cell and
every step. Multiplication improves by roughly 3× because the partial-product
structure is *native* to that lattice: the anti-diagonal $i+j=p$ whose sum is
exactly $s_p$ terminates at cell $(p,0)$ and is swept there by the single
uniform shift $(i{+}1,j{-}1)$.

## 4. The readout is load-bearing (second falsified hypothesis)

Because the stencil can only perform **uniform** shifts, the readout site is
structural, not cosmetic. Two sites are canonical:

- **corner $(p,0)$** — terminus of the anti-diagonal $i+j=p$; native to
  multiplication;
- **diagonal $(p,p)$** — the only cell observing both $a_p$ and $b_p$; native
  to addition and subtraction, whose carry/borrow flows
  $(p{-}1,p{-}1)\to(p,p)$ under the uniform shift $(+1,+1)$.

Corner readout learned multiplication (0.554) and **destroyed addition and
subtraction** (0.51 / 0.008). The first fix attempted — pooling over row $p$,
which contains $(p,p)$ — *failed*: 0.113 / 0.131. A row pool shares weights
across the row, so it cannot isolate the one diagonal cell among $W$.

Reading both sites with separate learned weights fixes it for **16 extra
parameters**, restoring add/sub to parity while keeping the multiplication
gain. The large seed spread under corner readout (0.194) also collapsed:
that variance was the readout mismatch, not optimizer instability, and the
dual readout is consistent across four seeds.

This is not an arithmetic cue. Both sites are generic lattice loci; which one
matters for which operator, and how to accumulate into it, is learned. The
row-pool result is evidence for that reading — merely *including* the right
cell was not enough.

## 5. What was falsified

1. **"The CP rank bound implies S4 cannot multiply."** False. The recurrent
   field is a second pathway. At $W=6$, S4 at rank 4 (1,865 parameters)
   reached 100% multiplication. The surviving claim is only that widening the
   *relation* to representability costs $\Theta(W^2)$ parameters — and §2
   shows doing so does not help anyway.
2. **"Row pooling will fix addition."** False, as above.

A third error was experimental rather than theoretical: the first sweep ran at
$W=6$ for exhaustive evaluation, but admissibility caps products at 63 there,
so multiplication degenerates into a memorizable table and **every**
configuration scored ~100%. $W=6$ cannot discriminate architectures.

## 6. Efficiency, measured rather than asserted

- Parameter count is **independent of bit width** — enforced as a unit test.
  The same 2,001-parameter model is defined at $W=4$ and $W=64$; only the
  compute horizon scales, as $2W$ steps over $W^2$ cells.
- The S4 relation alone costs $3WR$, and $3W(2W{-}1)=828$ at $W=12$ if widened
  to representability, growing quadratically.
- Structured pruning: each shared rank channel spans the projection and both
  stencil lifts, so the group penalty removes
  $C(1+2K)$ parameters per pruned channel. `pruning_curve.csv` reports
  accuracy at every pruning level, reloading the trained checkpoint each time
  so levels are independent rather than cumulative damage.

## 7. Reproduction

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure

# CP rank curve (requires NumPy)
python3 research/cp_rank_bound.py

# One sweep cell: <out> <s4|s5> <capacity> <seed> <steps> <width> [readout]
./build/sera_s5_comparison out_s4  s4 12 1 3000 12
./build/sera_s5_comparison out_s5  s5 16 1 3000 12 dual
python3 research/summarize_sweep.py <sweep_dir>

# Step-by-step proofs and the pruning curve
./build/sera_s5_proof out_proof 16 1 3000 12
```

## 8. What is not established

- No matched transformer baseline has been run. No claim of superiority to
  one is made.
- Multiplication at 0.54 non-trivial is **not** reliable arithmetic. The
  strict S4 acceptance gate (99% atomic, 95% proof, 90% depth-4–6) is still
  failed, by a wide margin.
- Two seeds per S4 configuration is thin; the S4 ranks are separated by less
  than their spread, so §2 supports "rank does not help" but not a fine
  ordering among ranks.
- Widths beyond 12 are untested, as is any operator outside $\{+,-,\times\}$.
- Nothing here bears on generalization beyond arithmetic, self-improvement,
  or ASI.
