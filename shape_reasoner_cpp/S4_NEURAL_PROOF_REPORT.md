# SERA S4 Neural Proof Research Report

## Verdict

**Acceptance gate: failed.**

The strongest checkpoint contains 4,785 parameters and genuinely learned
addition, subtraction, partial multiplication behavior, and reusable local
proof transitions from raw inputs. It is not reliable enough to call a
general arithmetic reasoner, much less ASI.

The failure is scientifically useful: increasing examples improved
in-domain multiplication but reduced magnitude generalization, while the same
frozen local error compounded sharply with proof depth.

## Closed-loop protocol

1. Generate valid unsigned 12-bit expression trees using an external label
   evaluator.
2. Train the same neural kernel on every true internal node.
3. Stratify exposure by the raw operator symbol; never dispatch arithmetic in
   the kernel.
4. Expand multiplication operands through \(1,3,7,15,31,63\).
5. Freeze parameters, save a checkpoint, reload it, and verify its checksum.
6. Evaluate atomic operands 0–63, magnitude-OOD operands 64–255, and proof
   trees of depths 1–6.
7. Record every atomic prediction and every proof step. Verify that evaluation
   causes zero parameter drift.

The final hard-example phase resumed the temporal model, used full-width
inputs immediately, and supplied 12 multiplication cases per batch while
retaining addition, subtraction, and tree-node examples.

## Architecture ablation

| Variant | Parameters | Dense reduction | Atomic ID | Magnitude OOD | Proof final | Semantic steps |
|---|---:|---:|---:|---:|---:|---:|
| Leaky 1-D field, rank 8 | 1,761 | 61.53% | 69.27% | 67.71% | 28.57% | 59.21% |
| Adaptive gate, rank 8 | 2,817 | 67.64% | 68.75% | 67.71% | 27.98% | 54.50% |
| CP relation + gate, rank 12 | 4,785 | 55.76% | 70.83% | 69.27% | 30.95% | 66.21% |
| + temporal supervision | 4,785 | 55.76% | 70.31% | 69.27% | 30.36% | 70.36% |
| + full-width hard phase | 4,785 | 55.76% | **75.52%** | 66.15% | **32.14%** | **72.50%** |

The gate alone did not help. The relational contraction and temporal loss
improved step learning. Extra full-width data then traded OOD generalization
for in-domain fit.

## Strongest checkpoint breakdown

| Split / operator | Exact accuracy |
|---|---:|
| In-domain addition | 100.00% |
| In-domain subtraction | 100.00% |
| In-domain multiplication | 26.56% |
| Magnitude-OOD addition | 93.75% |
| Magnitude-OOD subtraction | 96.88% |
| Magnitude-OOD multiplication | 7.81% |

| Proof depth | Final answer | Full proof | Semantic steps | Locally valid steps |
|---:|---:|---:|---:|---:|
| 1 | 72.92% | 72.92% | 72.92% | 72.92% |
| 2 | 27.08% | 27.08% | 52.08% | 72.22% |
| 3 | 6.25% | 4.17% | 42.86% | 71.13% |
| 4 | 12.50% | 0.00% | 61.67% | 80.00% |
| 5 | 12.50% | 12.50% | 85.48% | 93.15% |
| 6 | 12.50% | 12.50% | 94.25% | 97.42% |

The high local validity at small-value deep trees is not equivalent to
semantic correctness. A wrong child value can still be transformed
consistently at its parent. The report therefore keeps both metrics.

## Strict gate

The gate requires:

- atomic in-domain accuracy at least 99%;
- magnitude-OOD accuracy at least 95%;
- final and full-proof accuracy at least 95%;
- semantic and local step accuracy at least 99%;
- depth-4–6 final accuracy at least 90%;
- zero parameter drift during evaluation.

The checksum and immutability conditions pass. Every accuracy condition fails.

## Efficiency conclusions

What is proven:

- exact parameter count: 4,785;
- 55.76% fewer parameters than the defined dense analogue;
- one parameter set reused over positions, 20 recurrent steps, operators, and
  all proof nodes;
- 398,412 cumulative training cases recorded in the strongest checkpoint;
- approximately 0.68 ms per arithmetic node in the measured release build;
- checkpoint round-trip is bit-exact.

What is not proven:

- better sample efficiency than a transformer;
- better wall-clock or energy efficiency than a matched baseline;
- width extrapolation beyond the fixed 12-bit representation;
- reliable multiplication;
- wide-domain generalization, creativity, self-improvement, or ASI.

## Next falsifiable experiment

Do not merely increase rank. The next architecture should use a learned
two-dimensional interaction workspace with shared local transitions, then
compare it against:

1. this 4,785-parameter checkpoint;
2. a parameter-matched MLP;
3. a parameter-matched recurrent transformer;
4. a Neural-GPU-style dense gated convolution.

Use at least five independent seeds, exhaustive 6-bit atomic evaluation,
held-out operand widths, equal training examples, measured multiply-adds,
peak memory, latency, and calibration. Promote the design only if it clears
the same proof-depth gate without OOD regression.

## Reproduction

Build:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

Train the rank-12 model:

```bash
./build/sera_neural_proof_research results_s4 30000 - 0 12
```

Run the hard-example phase:

```bash
./build/sera_neural_proof_research results_s4_finetune 10000 \
  results_s4/sera_s4_proof_kernel.bin 0 12
```

The final directory contains `summary.json`, `model.json`,
`training_curve.csv`, `atomic_predictions.csv`, `proof_depth.csv`,
`proof_steps.csv`, `sample_proofs.json`, `input_policy.json`, and the binary
checkpoint.
