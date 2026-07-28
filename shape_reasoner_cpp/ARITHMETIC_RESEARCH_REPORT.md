# SERA Learned Arithmetic Research Report

Date: 2026-07-28  
Engine: C++20 only  
Compiler: GCC 13.3.0  
Release flags: `-O3 -march=native -DNDEBUG`

## Result

SERA learned three anonymous arithmetic relations from examples and used the
frozen models to solve unseen direct questions, extrapolated questions, inverse
questions, and two- or three-step expression shapes. Every genuine test answer
was correct. A permuted-label control collapsed to 3.11% accuracy.

This is a real learned-relation experiment, but it is deliberately narrow. It
shows autonomous symbolic factor discovery and composition inside a suitable
hypothesis class. It does not demonstrate general number sense or open-ended
mathematical intelligence.

## What “no hardcoding” means here

The arithmetic learner receives records of the form

```text
(anonymous operator ID, left value, right value, observed result)
```

It has no switch statement for addition, subtraction, or multiplication, no
operator symbols, no answer table, and no access to the evaluator. The
operator-specific arithmetic switch exists only in the separate research
harness, where it generates labeled examples and scores predictions after
inference. The learner source contains no reference to that function.

Some inductive bias is unavoidable. SERA is given a generic polynomial search
space containing every monomial whose total degree is at most three:

\[
\phi_{pq}(a,b)
  =\left(\frac{a}{s}\right)^p
   \left(\frac{b}{s}\right)^q,
\qquad p+q\le 3.
\]

It is not told which terms belong to which operator. Forward structure
selection chooses terms by held-out validation error, then ridge regression
fits their coefficients:

\[
\theta^\star
  =\arg\min_\theta
    \sum_n\left(
      \frac{y_n}{t}
      -\sum_{k\in S}\theta_k\phi_k(a_n,b_n)
    \right)^2
    +\lambda\|\theta\|_2^2.
\]

Therefore the honest statement is: no answers or operator rules were
hardcoded, but the generic low-degree polynomial hypothesis class was designed
by us.

## Dataset and isolation

- Operand training range: \([-8,8]\).
- Possible pairs per operator: 289.
- Operators: three anonymous IDs, displayed as `+`, `-`, and `*` only in
  reports.
- Training examples: 674.
- Frozen held-out examples: 193.
- Exact train/test pair overlap: 0.
- An additional internal fit/validation split selected model structure.
- Extrapolation tests used at least one operand outside \([-8,8]\), up to
  magnitude 16.
- The model was frozen before any reported evaluation.

Training order was independently shuffled and the model retrained. Maximum
held-out prediction drift was exactly zero.

## Discovered relation shapes

The coefficients below are converted back from normalized coordinates.
Coefficients smaller than \(10^{-12}\) are numerical zero.

| Anonymous operator | Selected nonzero terms | Discovered relation |
|---|---|---|
| 0 | `left`, `right` | \(f_0(a,b)\approx a+b\) |
| 1 | `left`, `right` | \(f_1(a,b)\approx a-b\) |
| 2 | `left*right` | \(f_2(a,b)\approx ab\) |

Raw fitted coefficients:

- operator 0: left \(0.9999999999999869\), right
  \(0.9999999999999872\);
- operator 1: left \(0.9999999999999880\), right
  \(-0.9999999999999886\);
- operator 2: left-right interaction \(0.9999999999999656\).

The learner autonomously rejected the unused squared and cubic terms.

## Shape inference

For an unseen direct question, every integer result candidate receives energy

\[
E(r)=\left(r-f_o(a,b)\right)^2.
\]

For an inverse question with unknown left operand,

\[
E(a)=\left(y-f_o(a,b)\right)^2.
\]

For the composed expression

\[
((a\ o_1\ b)\ o_2\ c)\ o_3\ d,
\]

SERA creates intermediate nodes \(z_1,z_2\) and a result node \(r\):

\[
E(z_1,z_2,r)=
  (z_1-f_{o_1}(a,b))^2+
  (z_2-f_{o_2}(z_1,c))^2+
  (r-f_{o_3}(z_2,d))^2.
\]

The existing categorical shape engine settles this chain with min-plus
messages. The arithmetic oracle never supplies an intermediate or final value
to this graph.

## Accuracy and operations

| Evaluation | Cases | Exact | Accuracy | Candidate evaluations |
|---|---:|---:|---:|---:|
| Disjoint held-out pairs | 193 | 193 | 100% | 99,009 |
| Unseen extrapolation | 352 | 352 | 100% | 180,576 |
| Unknown-left inverse problems | 180 | 180 | 100% | 5,940 |
| Two-step expression shapes | 60 | 60 | 100% | 3,810,789 |
| Three-step expression shapes | 60 | 60 | 100% | 9,368,883 |
| Permuted-label control | 193 | 6 | 3.11% | not used as SERA output |

All genuine tests had zero integer error. The negative control proves that
operator IDs, question order, or the scoring harness were not sufficient to
produce the result: destroying the learned relation destroyed accuracy.

## Representative autonomous outputs

| Problem | Test type | SERA | Expected |
|---|---|---:|---:|
| \(-7+4\) | held-out pair | -3 | -3 |
| \(9-14\) | extrapolated operands | -5 | -5 |
| \(-8\times7\) | held-out pair | -56 | -56 |
| \((3+7)\times4\) | two-step shape | 40 | 40 |
| \(((8-3)\times4)+7\) | three-step shape | 27 | 27 |
| \(?\times-4=24\) | inverse shape | -6 | -6 |

## Timing

- Relation discovery and fitting: 108.6 µs for 674 examples.
- Mean two-step shape inference: 222.0 µs.
- Mean three-step shape inference: 529.7 µs.
- Maximum observed three-step inference: 783.6 µs.

These are single-run measurements in a shared virtual sandbox, not controlled
hardware benchmarks.

## Autonomous controls and hardening

After launch, the executable autonomously:

1. generates and freezes train/test partitions;
2. verifies zero overlap;
3. discovers terms per anonymous operator;
4. fits and freezes coefficients;
5. evaluates held-out and extrapolated pairs;
6. constructs and settles inverse and composed shapes;
7. retrains after shuffling example order;
8. trains the permuted-label negative control;
9. emits every prediction and operation count; and
10. applies the pre-registered pass/fail contract.

The release binary exited successfully. A second build ran the same complete
protocol under AddressSanitizer and UndefinedBehaviorSanitizer with no reported
error. LeakSanitizer was disabled because this sandbox blocks the process
inspection it requires.

## Limitations

- The data is exact, synthetic, and noiseless.
- The chosen hypothesis family already contains the true relations.
- Addition, subtraction, and multiplication are unusually simple
  low-degree polynomials.
- Extrapolation is tested only to operand magnitude 16.
- Composition uses a bounded integer state domain \([-64,64]\).
- There is no division, remainder, decimal arithmetic, carry learning,
  natural-language parsing, or uncertainty in observations.
- Operators are learned independently after grouping by anonymous ID.
- This is structure discovery within a supplied representational language,
  not autonomous invention of mathematics from raw sensory experience.

The next falsification test should remove the hand-selected polynomial family:
give SERA a library of generic local programs or differentiable factor
templates, add division/remainder and noisy examples, and measure whether it
discovers compact executable relations without sacrificing negative-control
separation.

## Reproduction

```bash
g++ -std=c++20 -O3 -march=native -DNDEBUG \
  -Wall -Wextra -Wpedantic -Iinclude \
  src/shape_engine.cpp src/categorical_engine.cpp \
  src/arithmetic_learner.cpp src/arithmetic_research_main.cpp \
  -o sera_arithmetic

./sera_arithmetic results_arithmetic_final
```

The result directory contains all training examples, every prediction,
composition telemetry, inverse solutions, the frozen learned models, sample
outputs, and aggregate metrics.

