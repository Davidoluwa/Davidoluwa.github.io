# SERA Stage S0 — Equilibrium Substrate Report

## Outcome

**Stage S0 passed all pre-registered criteria.**

The C++ engine resolved vector-valued relational shapes without token-by-token
reasoning, produced order-invariant answers, extracted goal-conditioned
subshapes without changing those answers, and separated contradictory from
consistent evidence through residual energy.

This validates the numerical and telemetry substrate only. No representation,
factor, topology, or language component was learned.

## Stress design

The executable generated 288 deterministic cases spanning:

- relevant graph sizes of 8, 32, 128, and 256 nodes;
- one- and four-dimensional node states;
- tree dependencies plus redundant branch/merge chords;
- zero, one, or four disconnected distractor components;
- clean and deliberately contradictory evidence;
- six random seeds per configuration;
- shuffled factor order for every case.

The largest full shape contained 1,280 nodes and 1,705 factors. Every query was
solved both in the full shape and in the connected goal subshape.

## Pre-registered criteria and results

| Criterion | Threshold | Result | Status |
|---|---:|---:|---|
| Clean-case success rate | at least 99% | 100% | pass |
| Worst clean RMSE | at most 1.0e-3 | 2.46e-7 | pass |
| Worst factor-order output drift | at most 1.0e-4 | 0 | pass |
| Worst goal-slice query drift | at most 1.0e-4 | 2.38e-7 | pass |
| Conflict/clean energy ratio | at least 5 | 3.39e11 | pass |
| Factor reduction with 4x distractors | at least 70% | 80% | pass |

The machine-readable decision is in
[`results_s0_final/aggregate.json`](results_s0_final/aggregate.json).
Per-case evidence is in
[`results_s0_final/stress_summary.csv`](results_s0_final/stress_summary.csv).

## CPU behavior

Across the 144 clean release-build cases:

| Statistic | Solve time |
|---|---:|
| Mean | 684 us |
| Median | 156 us |
| 95th percentile | 2.81 ms |
| 99th percentile | 6.53 ms |
| Maximum | 7.23 ms |

The mean solve required 66.7 equilibrium iterations; the maximum was 147. These
numbers include complete full-shape solving but not file output.

The current implementation is deterministic and single-threaded. Its reasoning
semantics are parallel/global—the whole factor field defines each Krylov
direction—but S0 does not yet parallelize the factor kernels across CPU cores.

## Closed-loop failure and revision

The first run produced highly accurate states but labelled only 41.7% of clean
cases converged. The requested linear residual was below the meaningful float32
state precision, so conjugate-gradient directions numerically vanished before
the telemetry threshold was crossed.

An initial precision-aware early stop repaired the convergence label but
degraded worst goal drift to 1.78e-3, violating the contract. The correct
revision was to:

1. continue solving toward the requested tolerance;
2. use the float32 precision floor only when classifying numerical exhaustion;
3. report requested and effective precision limits separately.

The final run restored the 2.38e-7 goal drift and reached 100% clean-case
success. This sequence is retained because it demonstrates why telemetry must
distinguish mathematical convergence, representational precision, and answer
accuracy.

## Sanitizer validation

The engine was rebuilt with AddressSanitizer and UndefinedBehaviorSanitizer and
reran all 288 cases successfully. LeakSanitizer alone could not inspect `/proc`
under the sandbox's tracing restrictions, so leak probing was disabled for the
successful sanitizer pass.

## What S0 establishes

1. A relational shape can serve as the direct object of computation.
2. Matrix-free equilibrium solving is insensitive to factor presentation order.
3. A goal-conditioned causal component can remove irrelevant structure without
   materially changing the answer.
4. Contradictions remain visible as residual energy rather than being silently
   overwritten by a sequential answer.
5. A settled shape can be compiled into a compact semantic skeleton with answer,
   support, contradiction, and provenance fields.

## What S0 does not establish

1. The engine did not discover nodes, factors, or goals.
2. The tested factors were linear and convex.
3. Goal slicing was deterministic and used explicit topology.
4. No ambiguous competing topologies or particles were tested.
5. No local learning rule was exercised.
6. No raw perception or language entered the engine.
7. A sparse linear solver is not itself a new intelligence architecture.

The architecture becomes scientifically interesting only if later stages can
learn useful shapes and retain these computational properties.

## Next falsifiable stage

Stage S1 should introduce categorical beliefs, nonlinear factors, and competing
shape particles. It should test whether whole-shape settlement beats sequential
commitment under ambiguity and backtracking while tracking:

- accuracy and calibration;
- local-minimum failures;
- particle diversity and collapse;
- compute per reduced uncertainty;
- sensitivity to factor order;
- contradiction localization;
- end-to-end selector and solver cost.
