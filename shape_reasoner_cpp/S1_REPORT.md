# SERA S1 Hardened Research Report

Date: 2026-07-28  
Implementation: C++20 only  
Compiler: GCC 13.3.0, `-O3 -march=native -DNDEBUG`  
Sandbox: Linux 6.12.13 x86-64, KVM, AMD EPYC 9V74, 9 visible CPUs,
15 GiB RAM

## Executive result

S1 passes its bounded research contract. It forms and settles categorical
relational shapes without selecting tokens, records its primitive
counterfactual work, preserves complete correlated alternatives with adaptive
particles, and emits a compact assignment/confidence/contradiction object for
a later renderer.

The strongest result is not that the first solver was perfect. It was not.
Exhaustive enumeration exposed one cyclic MAP failure. The non-converged
message field triggered a second shape split, and depth-2 particle conditioning
recovered the exact optimum. This is the desired closed-loop behavior:
telemetry changed the inference topology instead of being decorative logging.

This remains a substrate result, not evidence of human-level reasoning. The
tests use supplied graph structure and factors; S1 does not learn them from raw
experience.

## What changed from S0

### S0 hardening

- Removed a per-trace allocation from the PCG settlement loop.
- Corrected workspace accounting.
- Rejected non-finite targets, zero/non-finite coefficients, duplicate node
  references inside one linear factor, empty node types, and invalid solver
  options.
- Added explicit termination reasons for convergence, precision exhaustion,
  iteration limit, and numerical breakdown.
- Re-ran all 288 S0 stress cases.

### S1 kernel

- General categorical nodes with arbitrary finite unary costs.
- General pairwise factors with dense \(K_i\times K_j\) cost tables.
- Damped min-plus counterfactual factor messages.
- Deterministic synchronous active-frontier settlement.
- Exact goal-component extraction before inference.
- Confidence and entropy readout separated from settlement.
- Adaptive correlated shape particles with finite clamps, recursive splitting,
  beam budgeting, original-energy scoring, and mode deduplication.
- Exact primitive-operation telemetry.
- Categorical assignment/confidence/contradiction JSON compiler.

The derivation is in `S1_MATHEMATICS.md`.

## Final release results

| Test | Result | Contract |
|---|---:|---:|
| Anchored modular shapes | 36/36 solved | at least 98% |
| Factor-order assignment failures | 0/36 | 0 |
| Worst order energy drift | 0 | at most \(10^{-9}\) |
| Goal-slice query failures | 0/36 | 0 |
| Factor reduction with four distractor components | 80% | at least 79% |
| Contradiction separations | 36/36 | all |
| Minimum contradiction energy ratio | \(2.5\times10^{13}\) | at least 10 |
| Canonical ambiguity particle cases | 6/6 full mode coverage | all |
| Random tree exact-oracle MAP | 24/24 | 24/24 |
| Random loopy base MAP | 23/24 | diagnostic |
| Random loopy MAP after particles | 24/24 | 24/24 |
| Operation-accounting failures | 0/36 | 0 |
| Weighted active-frontier candidate reduction | 82.30% | at least 50% |
| Malformed-input checks | 5/5 | 5/5 |
| S0 regression | 288/288 clean suite status | pass |
| ASan + UBSan S1 rerun | no reported error | pass |

The weighted 82.30% reduction compares actual \(K^2\) counterfactual
evaluations with dense evaluation of every directed message on every executed
sweep. Across individual clean cases, the reduction ranged from 10.93% to
89.88%, with a 58.48% unweighted mean. It does not claim an 82% reduction in
total CPU instructions.

## Timing

Release timing for the 36 clean anchored cases:

| Statistic | Time |
|---|---:|
| Mean | 536.8 µs |
| p50 | 256.3 µs |
| p95 | 2.239 ms |
| p99 | 2.242 ms |

The largest tested clean shape had 960 nodes and 1,275 factors. One such
two-state case settled in 63 sweeps and 1.530 ms using 105,900 bytes of
reported contiguous numeric workspace.

These are quick sandbox measurements, not a controlled hardware benchmark.
The virtual machine is shared, only one release run supplies the reported
timings, and no CPU affinity or frequency control was available.

## Primitive operation ledger

The solver counts algorithmic primitives in its controlled numeric path:

- directed message proposals, commits, and gated skips;
- \(K_iK_j\) counterfactual candidate evaluations;
- scalar additions/subtractions, comparisons, and multiplies;
- exponentials used only for output uncertainty;
- final factor-energy lookups.

It intentionally excludes allocator internals, `std::vector` bookkeeping,
hash byte operations, clocks, file I/O, JSON formatting, and compiler-generated
machine instructions. For homogeneous-\(K\) stress cases, the harness checks

\[
\text{candidate evaluations}
  =K^2\times\text{message proposals}
\]

and

\[
\text{message proposals}
  =\text{commits}+\text{gated skips}.
\]

All 36 accounting checks passed.

### Representative clean operation record

Case `n16_k3_d4_s17` contains 80 nodes, 105 factors, one 16-node goal component,
and four disconnected distractor components.

| Field | Value |
|---|---:|
| Sweeps | 27 |
| Final energy | 0 |
| Violated factors | 0 |
| Message proposals | 1,178 |
| Message commits | 812 |
| Gated skips | 366 |
| Counterfactual candidates | 10,602 |
| Scalar add/subtracts | 50,406 |
| Scalar comparisons | 13,678 |
| Scalar multiplies | 4,254 |
| Output exponentials | 240 |
| Factor-energy lookups | 105 |

A dense schedule over the same 27 sweeps would propose
\(2(105)(27)=5{,}670\) messages and evaluate 51,030 candidates. The active
frontier therefore removed 79.22% of candidate evaluations in this example.

Settlement telemetry:

| Recorded sweep | Max residual | Total residual | Provisional energy | Commits | Gated skips |
|---:|---:|---:|---:|---:|---:|
| 1 | 18 | 54 | 325 | 3 | 207 |
| 27 | \(8.0474\times10^{-9}\) | \(1.14785\times10^{-7}\) | 0 | 0 | 21 |

The trace is sampled every 32 sweeps plus termination, so the unlisted sweeps
still appear in aggregate operation counts.

## Concrete outputs

### Clean compiled shape

For the representative case, the 16-node goal component compiled to:

```text
[2, 0, 0, 2, 2, 1, 1, 1, 0, 0, 2, 0, 1, 2, 2, 0]
```

Every goal-node confidence was 1.0, final energy was zero, and no factor was
violated. The four unanchored equality distractor components remained
explicitly uncertain at confidence \(1/3\); they were not falsely described as
known.

### Contradictory compiled shape

After corrupting one relation in the same case:

| Field | Clean | Contradictory |
|---|---:|---:|
| Sweeps | 27 | 39 |
| Energy | 0 | 25 |
| Violated factors | 0 | 1 |
| Counterfactual candidates | 10,602 | 12,456 |

The engine retained the best global assignment but surfaced the irreducible
factor conflict instead of fabricating a consistent explanation.

### Correlated ambiguity

Unanchored modular rings have \(K\) globally correlated zero-energy modes. The
base marginal field correctly had entropy \(\log K\). Across two seeds for
each \(K\in\{2,3,5\}\), particle splitting recovered all \(K\) unique modes
with zero original energy and zero constraint violations.

For \(K=5\), each case used five particle solves and recovered five modes in
about 0.43–0.45 ms.

## Autonomous failure-and-repair loop

The development run deliberately progressed through falsifiable checks:

1. The first S1 implementation passed planted constraints but still proposed
   every stable message.
2. An operation audit rejected that as merely write-gated. The solver was
   changed to a causal active frontier.
3. The frontier retained all correctness checks and reduced weighted
   counterfactual candidates by 82.30%.
4. Exhaustive enumeration on 48 small random problems then exposed one loopy
   failure: `loop_k3_s6` did not converge by 640 sweeps and returned energy
   100.38000753 instead of the exact 80.17000729.
5. One-level particle conditioning improved that case to 80.58000725 but
   remained 0.40999996 above optimum.
6. Telemetry-triggered depth-2 conditioning recovered energy 80.17000729
   exactly. The final bounded oracle result is 48/48 with adaptive particles.

This does not turn loopy belief propagation into a generally exact polynomial
algorithm. Recursive conditioning is exponential in the worst case. The result
shows that convergence and entropy telemetry can productively allocate a small,
explicit branching budget.

## Hardening evidence

The malformed-input suite verified rejection of:

1. a non-finite S0 target;
2. duplicate S0 factor references to the same node;
3. parallel S1 pairwise factors that should be fused;
4. a non-finite S1 unary cost; and
5. invalid zero damping.

The ASan/UBSan build re-ran the complete S1 harness without a sanitizer report.
LeakSanitizer was disabled because the sandbox blocks the process inspection it
requires; this is the same environmental limitation recorded for S0. ASan
heap/stack checks and UBSan remained enabled.

S0’s final regression reproduced:

- 288 cases;
- 100% clean success;
- maximum clean RMSE \(2.45756\times10^{-7}\);
- zero factor-order drift;
- maximum goal-slice query drift \(2.38419\times10^{-7}\);
- contradiction energy ratio \(3.3923\times10^{11}\); and
- 80% mean factor reduction with four distractor components.

## Research positioning

The implementation is genuinely experimental, but its ingredients must be
named accurately. Factor graphs and message passing are established
([Kschischang et al.](https://ieeexplore.ieee.org/document/910572/));
residual scheduling is established
([Elidan et al.](https://arxiv.org/abs/1206.6837)); particle belief propagation
is established
([Ihler and McAllester](https://proceedings.mlr.press/v5/ihler09a.html)); and
tree-reweighted methods are an important established route to more controlled
loopy inference
([Kolmogorov](https://proceedings.mlr.press/r5/kolmogorov05a/kolmogorov05a.pdf)).

The SERA research candidate is their CPU-first composition around an explicit
non-linguistic shape:

- the topology is the thought object;
- counterfactual tension is the compute signal;
- causal settlement precedes communication;
- uncertainty and non-convergence trigger correlated structural alternatives;
- exact operation telemetry makes compute falsifiable; and
- language receives only the compiled result.

## Limitations and next falsification target

- All test topology and factor semantics are provided by the harness.
- Only finite pairwise categorical factors are implemented.
- The particle oracle is bounded to tiny graphs; it is not a general proof.
- The test distribution emphasizes modular constraints and small random energy
  tables, not grounded tasks.
- Goal extraction is exact connected-component selection, not learned
  relevance.
- No temporal memory, identity learning, topology editing, factor learning, or
  natural-language parser is present.
- Timing is not compared with a strong exact solver such as branch-and-bound,
  dynamic programming on bounded treewidth, ILP, or TRW-S.

The next serious stage should learn or infer relevance and factor structure
from non-linguistic transition data, while retaining an exact small-problem
oracle and adding TRW-S or branch-and-bound as a certified comparator. If SERA
cannot recover topology without hand-coded relations, this architecture should
be materially revised rather than cosmetically scaled.

## Reproduction

```bash
g++ -std=c++20 -O3 -march=native -DNDEBUG \
  -Wall -Wextra -Wpedantic -Iinclude \
  src/shape_engine.cpp src/categorical_engine.cpp src/s1_stress_main.cpp \
  -o sera_s1

./sera_s1 results_s1_final
```

The result directory contains raw case rows, relational checks, exact-oracle
comparisons, particle records, compiled sample outputs, and settlement traces.

