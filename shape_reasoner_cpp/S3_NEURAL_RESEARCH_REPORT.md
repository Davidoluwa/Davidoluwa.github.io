# SERA S3 Neural Shape Research Report

Date: 2026-07-28  
Implementation: C++20  
Final architecture: recurrent neural cellular field  
Transformer attention: none

## Executive result

SERA now has reusable neural parameters and still reasons by constructing a
whole relational shape before answering.

The final checkpoint contains 1,993 learned parameters. It was meta-trained
once over 8,000 anonymous episodes, reloaded into a new model object, and
frozen throughout fresh confirmation. It received no operation identifier,
operand grouping, bit significance, wire permutation, arithmetic primitive, or
confirmation feedback.

On the untouched confirmation set:

| Metric | Result |
|---|---:|
| Fresh arithmetic episodes | 96 |
| Withheld six-bit outputs | 384 |
| Exact outputs | 349/384 |
| Exact accuracy | 90.885% |
| Bit accuracy | 97.179% |
| Addition exact accuracy | 124/128 (96.875%) |
| Subtraction exact accuracy | 117/128 (91.406%) |
| Multiplication exact accuracy | 108/128 (84.375%) |
| Permuted-output control | 27/384 (7.031%) |
| Random-weight architecture | 41/384 (10.677%) |
| Support-order drift | 0 |
| Parameter drift during evaluation | 0 |
| Checkpoint reload mismatch | 0 |

The optimized and ASan/UBSan builds produced byte-identical prediction files.

## What changed from S2

S2 solved a new GF(2) program search independently in every episode. S3 learns
one shared neural dynamical law and uses it to construct a new episode shape
without changing its parameters.

| Property | S2 symbolic ANF | S3 neural shape field |
|---|---:|---:|
| Reusable learned parameters | none | 1,993 |
| Per-episode gradient updates at confirmation | none | none |
| Final exact accuracy | 84.375% | 90.885% |
| Final bit accuracy | 95.660% | 97.179% |
| Multiplication exact accuracy | 65.625% | 84.375% |
| Mean episode construction | 43.28 µs | 4.67 ms |
| Query execution | ANF parity | frozen-field lookup |
| State scaling | exponential | exponential |

S3 improves overall exact accuracy by 6.510 percentage points and
multiplication by 18.750 points. The price is approximately 108 times more
episode-construction compute than S2. Its advantage is reusable knowledge and
more balanced generalization, not cheaper adaptation.

## Sparse-support sweep

The same frozen global parameters were tested with different support sizes:

| Support states | Domain observed | Exact accuracy | Bit accuracy |
|---:|---:|---:|---:|
| 8 | 12.5% | 14.918% | 68.536% |
| 16 | 25.0% | 22.352% | 73.506% |
| 24 | 37.5% | 32.786% | 78.954% |
| 32 | 50.0% | 42.188% | 83.968% |
| 40 | 62.5% | 56.163% | 88.484% |
| 48 | 75.0% | 70.182% | 92.578% |
| 56 | 87.5% | 86.719% | 96.419% |
| 60 | 93.75% | 90.885% | 97.179% |

This is a real adaptation curve rather than a single selected support size. It
also shows that S3 is not yet strongly sample efficient: high exact accuracy
still requires most of the 64-state domain.

## Reusable-parameter evidence

Three controls separate parameter learning from support copying:

1. The final checkpoint checksum was
   `1693126785264812350` before and after all confirmation episodes.
2. Reversing every support set caused zero prediction drift.
3. The same architecture with random weights achieved only 10.677% exact
   accuracy, versus 90.885% after meta-training.

During meta-training, the network input consumed 286,744 support
demonstrations across 8,000 episodes. Complete anonymous relations were used
as supervised reconstruction targets. This is substantial prior training and
must not be confused with 60-example learning from scratch.

## Unseen relation families

No additional training was performed for these evaluator-side relations:

| Unseen evaluator relation | Exact accuracy | Bit accuracy |
|---|---:|---:|
| Bitwise XOR | 100% | 100% |
| Bitwise AND | 100% | 100% |
| Bitwise OR | 100% | 100% |
| Affine composition | 82.813% | 97.135% |
| Multiply-accumulate | 36.719% | 79.167% |
| Nonlinear composition | 42.969% | 87.630% |
| Combined | 592/768 (77.083%) | 93.989% |

This demonstrates transfer but not wide compositional generalization. The
model transfers strongly to locally regular Boolean fields and poorly to novel
multiplicative compositions.

## Shape reasoning mechanism

The support set initializes a 64-node Boolean-state graph for each anonymous
output wire. A shared recurrent neural cellular rule propagates evidence for
six steps. The complete final activation tensor has shape:

\[
6\times64\times24.
\]

This tensor is frozen before any query is answered. The system therefore does
not discover its reasoning while generating an output sequence. It constructs
one episode-level latent object and then reads answers from it.

The detailed equations are in `S3_NEURAL_ARCHITECTURE.md`.

## Closed-loop refinements

1. **S2 baseline:** 84.375% exact; multiplication 65.625%.
2. **Initial 500-step neural field:** 28.646% exact.
3. **4,000-step field:** 82.813% exact.
4. **8,000-step field with derived Hamming feature:** 89.583% on fresh
   confirmation.
5. **Policy correction:** the derived Hamming feature was rejected even
   though it was relation-neutral.
6. **Raw-wire-only retraining:** 91.667% on development confirmation B.
7. **Compute correction:** cached neighbor aggregation reduced mean
   compilation from 9.18 ms to 4.47 ms with byte-identical predictions.
8. **Harder unseen-family suite:** exposed weak multiplicative composition
   instead of reporting only the perfect bitwise result.
9. **Untouched confirmation C:** 90.885% exact with all source, thresholds,
   checkpoint, and relation suites locked.

All intermediate failures and superseded checkpoints are retained.

## Hardening

- C++20 implementation with no Python runtime.
- Input and output values and dimensions are validated.
- Conflicting duplicate support entries are rejected.
- Checkpoint magic, version, tensor sizes, and truncation are validated.
- A deliberately truncated checkpoint exited with an error.
- The checkpoint was loaded into a separately initialized model.
- Parameters remained frozen throughout confirmation.
- Release and sanitizer prediction files are identical.
- ASan and UBSan reported no error.
- LeakSanitizer was disabled because process inspection is restricted in the
  sandbox.

## Efficiency and scaling

The reusable parameter footprint is excellent:

- 1,993 float parameters;
- approximately 7.8 KiB checkpoint;
- weights independent of episode count; and
- constant-time query lookup after shape compilation.

The workspace scaling is not:

\[
V=2^n,\qquad
\text{compile}=O(mT2^n(H^2+nH)).
\]

S3 is therefore a valid neural proof of the shape-before-answer mechanism, but
not a scalable wide-domain architecture. S4 should replace the complete
Boolean cube with a sparse dynamically constructed graph, share discovered
subshapes across domains, and learn which latent nodes to instantiate.

## Conclusion

S3 clears the requested next step:

- it is neural;
- it is not transformer-style;
- its learned parameters are reusable;
- it receives no operation cue;
- it constructs a complete latent shape before answering; and
- it materially outperforms S2 on fresh arithmetic episodes.

The result should not be called general intelligence. Dense state enumeration,
heavy meta-training, and weak unseen multiplicative composition remain the
central limitations.
