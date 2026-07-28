# SERA S2 Strict No-Cue Research Report

Date: 2026-07-28  
Implementation: C++20 only  
Final confirmation: 96 fresh anonymous episodes  
Sanitizers: AddressSanitizer and UndefinedBehaviorSanitizer

## Executive result

SERA passed the strict no-cue acceptance gate, but with a clear unresolved
weakness in multiplication.

The final learner received only scrambled Boolean input/output demonstrations.
It received no integers, operand grouping, bit significance, operator identity,
symbols, arithmetic primitives, polynomial features, wire significance, or
test feedback. It induced generic Boolean programs, froze them, and predicted
four withheld patterns per episode.

Final fresh-confirmation result:

| Metric | Result | Gate |
|---|---:|---:|
| Anonymous episodes | 96 | — |
| Demonstrations | 5,760 | — |
| Withheld queries | 384 | — |
| Train/test overlap | 0 | 0 |
| Training bit errors | 0 | 0 |
| Training-order output drift | 0 | 0 |
| Exact six-bit outputs | 324/384 | at least 80% |
| Exact-word accuracy | 84.375% | at least 80% |
| Bit accuracy | 95.660% | at least 95% |
| Permuted-output word accuracy | 4.427% | below 10% |
| Release versus sanitizer result | identical | identical |

This is genuine cue-free induction inside a small, bounded Boolean world. It is
not yet robust cue-free arithmetic in the general sense.

## Strict input policy

The complete policy is in `NO_CUE_POLICY.md`. The only learner-visible objects
were:

1. an episode boundary;
2. 60 pairs of scrambled six-bit input and six-bit output vectors; and
3. four scrambled six-bit queries after freezing.

An episode boundary is necessary. If examples from several incompatible
functions are mixed without any indication of which mapping applies, no
learner can infer a unique output for the same input.

Every episode independently permuted all input and output wires. Consequently,
the learner could not exploit a fixed operand boundary, least-significant bit,
carry direction, or output magnitude.

The semantic labels `addition_mod64`, `subtraction_mod64`, and
`multiplication_mod64` exist only in evaluator reports. They never enter the
learner API.

## Refined mathematics

### Initial failure: clause covers

The first system induced minimum-description DNF/CNF clause programs.
Although it fit every demonstration, it achieved only:

- 48.18% exact six-bit outputs; and
- 83.85% bit accuracy.

Parity and carry are expensive in pure clause form. The learner created broad
clauses through unknown truth-table regions and generalized poorly.

### Canonical Boolean algebra

The refined learner uses algebraic normal form:

\[
f_j(x)=
\bigoplus_{S\subseteq\{1,\ldots,6\}}
c_{j,S}\prod_{i\in S}x_i,
\qquad c_{j,S}\in\mathrm{GF}(2).
\]

For each output wire \(j\), every demonstration supplies one linear equation
over \(\mathrm{GF}(2)\):

\[
A c_j=y_j.
\]

Gaussian elimination gives a particular solution \(c_0\) and a nullspace basis
\(\{n_k\}\). Every consistent program is

\[
c=c_0\oplus\bigoplus_k\alpha_kn_k.
\]

With 60 independent demonstrations over 64 possible inputs, the typical
version space has dimension four. SERA enumerates the 16 consistent programs
per output wire and minimizes description length:

\[
L(c)=\sum_{S:c_S=1}
\left[2+|S|+\max(|S|-1,0)\right].
\]

This penalizes extra monomials and high-order conjunctions without containing
an arithmetic-specific rule.

### Refinement history

| Development stage | Withheld patterns/episode | Word accuracy | Bit accuracy | Decision |
|---|---:|---:|---:|---|
| DNF/CNF only | 16 | 48.18% | 83.85% | rejected |
| Mixed clauses and ANF | 16 | 72.14% | 91.49% | rejected |
| Canonical ANF | 4 | 90.63% | 98.09% | retained |
| Fresh confirmation set A | 4 | 86.46% | 96.74% | passed |
| Joint multi-output MDL | 4 | 87.24% | 96.96% | rejected for cost |
| Fresh confirmation set B | 4 | 84.38% | 95.66% | final |

Joint multi-output compression recovered only three additional words on set A
while increasing mean training time from approximately 46 µs to 5.24 ms—over
100 times more compute. The independent ANF solver is therefore the retained
Pareto point.

## Per-relation result

The learner did not see these names; they are evaluator-side interpretations.

| Evaluator relation | Exact words | Word accuracy | Bit accuracy |
|---|---:|---:|---:|
| Addition modulo 64 | 128/128 | 100% | 100% |
| Subtraction modulo 64 | 112/128 | 87.5% | 93.75% |
| Multiplication modulo 64 | 84/128 | 65.625% | 93.229% |

Addition is completely identified by the minimum-description ANF across all
fresh episodes. Subtraction is strong but variable. Multiplication remains
underdetermined: several consistent programs have similar description length
but disagree on the withheld inputs.

The overall pass must not obscure that multiplication would fail an 80%
per-relation requirement.

## Negative control

The training outputs were randomly permuted within each anonymous episode,
then passed through the identical learner and evaluator.

| Evaluator-side relation | Exact negative-control words |
|---|---:|
| Addition | 8/128 |
| Subtraction | 8/128 |
| Multiplication | 1/128 |
| Combined | 17/384 (4.427%) |

Random relations still admit programs that perfectly fit the 60 demonstrations,
but they do not predict the original withheld mapping. This strongly separates
learned input/output structure from evaluator leakage.

## Representative withheld outputs

The following semantic decoding occurred only after prediction:

| Evaluator interpretation | Predicted | Expected | Correct |
|---|---:|---:|---|
| \(2+7\) | 9 | 9 | yes |
| \(7+5\) | 12 | 12 | yes |
| \(6-1\) | 5 | 5 | yes |
| \(2-6\pmod{64}\) | 60 | 60 | yes |
| \(6\times7\) | 50 | 42 | no |
| \(7\times3\) | 29 | 21 | no |

Failures are deliberately included. Both multiplication predictions fit all 60
demonstrations in their episodes; the missing patterns expose ambiguity in the
chosen inductive bias.

## Autonomy

After process launch the system autonomously:

1. created anonymous episodes;
2. scrambled every input and output wiring;
3. separated demonstrations from withheld queries;
4. verified zero overlap;
5. formed and solved the GF(2) coefficient system;
6. enumerated the consistent version space;
7. selected minimum-description programs;
8. froze and executed them;
9. retrained after example-order shuffling;
10. ran 96 permuted-output controls;
11. recorded all successes and failures; and
12. applied the acceptance gate.

No human selected a program, repaired a prediction, or supplied a cue during
the confirmation run.

## Efficiency and operations

- Mean release training time: 43.28 µs per episode.
- Typical consistent programs evaluated: \(16\times6=96\) per episode.
- Total selected input literals across 96 six-output models: 11,087.
- Clause-cover combination work in the final solver: zero.
- Prediction is a parity reduction over the selected monomials.

The rejected joint model cost approximately 5.24 ms per episode for less than
one percentage point of extra exact accuracy.

## Hardening

- Duplicate Boolean inputs with conflicting outputs are rejected.
- Input/output dimensions and bit values are validated.
- The ANF version-space search is explicitly bounded.
- Training reproduction is independent of example ordering.
- Release and sanitizer binaries produced identical metrics.
- ASan and UBSan reported no error.
- LeakSanitizer was disabled because process inspection is restricted in this
  sandbox.

## Interpretation

This result is stronger than the earlier polynomial arithmetic experiment:

- the learner receives no decoded numbers;
- it receives no operator ID;
- it receives no operand boundary;
- wires change meaning in every episode;
- it receives no arithmetic hypothesis family; and
- the confirmation wiring and splits were fresh.

The evidence supports autonomous relation induction and minimum-description
program selection in a bounded Boolean domain. It does not support a claim of
general arithmetic intelligence. Sixty of 64 possible inputs are demonstrated,
and multiplication generalization is still weak.

The next mathematical target is a generic shared-subcircuit learner that can
discover reusable latent gates without the 100× overhead of the rejected joint
beam. It should be evaluated at 48/64 demonstrations, where the current system
fails, and it must improve multiplication without receiving operand or
bit-significance cues.

## Reproduction

```bash
g++ -std=c++20 -O3 -march=native -DNDEBUG \
  -Wall -Wextra -Wpedantic -Iinclude \
  src/boolean_program_learner.cpp src/no_cue_research_main.cpp \
  -o sera_no_cue

./sera_no_cue results_no_cue_final_v2
```

The result directory contains the enforced input policy, all 384 withheld
predictions, all episode metrics, negative controls, representative frozen
programs, and decoded sample outputs.
