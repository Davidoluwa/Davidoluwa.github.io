# SERA C++ Research Engine

SERA is an experimental non-token reasoning substrate. It represents a problem
as a typed relational shape, settles the whole shape to an equilibrium, and
then compiles the result into a semantic skeleton.

This repository contains the hardened S0 substrate and the tested S1
categorical kernel:

- generic vector-valued linear relational factors;
- matrix-free preconditioned conjugate-gradient equilibrium solving;
- general categorical pairwise factors and min-plus counterfactual messages;
- deterministic residual-gated active-frontier settlement;
- adaptive correlated shape particles;
- deterministic goal-conditioned subshape extraction;
- contradiction and convergence telemetry;
- semantic-skeleton and categorical JSON output;
- exact-enumeration oracle tests on small tree and loopy shapes;
- C++ stress generation across size, factor order, distractors, conflicting
  evidence, ambiguity, and malformed inputs.
- a learned-arithmetic experiment that autonomously selects sparse polynomial
  factor structure and composes the frozen relations as SERA shapes.
- a strict no-cue Boolean induction experiment with scrambled wires,
  minimum-description GF(2) program discovery, fresh confirmation episodes,
  and negative controls.
- a reusable non-transformer recurrent neural shape field that compiles
  anonymous demonstrations into a frozen whole-relation representation.
- an S4 CP-relational gated neural proof kernel that reuses 4,785 parameters
  across recurrent time and expression-tree nodes, emits explicit arithmetic
  proof ledgers, and intentionally fails closed when strict accuracy gates are
  not met.
- an S5 shift-equivariant 2-D interaction workspace that replaces the
  factorized S4 relation with an exact bilinear lattice and a single tied
  stencil, giving a parameter count independent of bit width, with a
  finite-difference-verified backward pass and structured rank-channel
  pruning. See `S5_INTERACTION_WORKSPACE.md` and `S5_RESULTS.md`.

The engine itself contains no language model, tokenizer, Python, external ML
framework, or claim of learned grounding. The `research/` directory holds
offline analysis scripts that do use Python and NumPy; nothing under `src/` or
`include/` depends on them.

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

Or compile directly with GCC/Clang:

```bash
g++ -std=c++20 -O3 -march=native -Wall -Wextra -Wpedantic \
  -Iinclude src/shape_engine.cpp src/stress_main.cpp -o sera_stress

g++ -std=c++20 -O3 -march=native -Wall -Wextra -Wpedantic \
  -Iinclude src/shape_engine.cpp src/categorical_engine.cpp \
  src/s1_stress_main.cpp -o sera_s1_stress

g++ -std=c++20 -O3 -march=native -Wall -Wextra -Wpedantic \
  -Iinclude src/shape_engine.cpp src/categorical_engine.cpp \
  src/arithmetic_learner.cpp src/arithmetic_research_main.cpp \
  -o sera_arithmetic_research

g++ -std=c++20 -O3 -march=native -Wall -Wextra -Wpedantic \
  -Iinclude src/boolean_program_learner.cpp src/no_cue_research_main.cpp \
  -o sera_no_cue_research

g++ -std=c++20 -O3 -march=native -Wall -Wextra -Wpedantic \
  -Iinclude src/neural_shape_field.cpp src/neural_shape_research_main.cpp \
  -o sera_neural_shape_research

g++ -std=c++20 -O3 -march=native -Wall -Wextra -Wpedantic \
  -Iinclude src/neural_proof_kernel.cpp src/neural_proof_research_main.cpp \
  -o sera_neural_proof_research
```

## Run

```bash
./build/sera_stress results
./build/sera_s1_stress results_s1
./build/sera_arithmetic_research results_arithmetic
./build/sera_no_cue_research results_no_cue
./build/sera_neural_shape_research results_neural_shape 8000
./build/sera_neural_proof_research results_s4 30000 - 0 12
```

Generated artifacts:

- `stress_summary.csv`: one row per case;
- `aggregate.json`: success metrics;
- `iteration_trace.csv`: convergence trace for a representative case;
- `sample_skeleton.json`: high-level compiled output.

S1 additionally emits:

- `cases.csv`: every solve and its primitive operation ledger;
- `relational_checks.csv`: factor-order, slicing, contradiction, and operation
  reconciliation checks;
- `particles.csv`: ambiguity mode coverage and particle work;
- `exact_oracle.csv`: inferred versus exhaustively optimal MAP energy;
- `sample_*`: exact settlement trace and compiled outputs;
- `summary.json`: final aggregate metrics.

The research contracts are `mission.json` and `mission_s1.json`. The S1
derivation is in `S1_MATHEMATICS.md`; measured results and limitations are in
`S1_REPORT.md`.

The learned arithmetic contract is `mission_arithmetic.json`; its full
protocol, outputs, anti-leakage controls, and limitations are in
`ARITHMETIC_RESEARCH_REPORT.md`.

The strict cue policy is `NO_CUE_POLICY.md`; the final experiment and complete
failure/refinement history are in `NO_CUE_RESEARCH_REPORT.md`.

The reusable neural policy is `S3_NEURAL_SHAPE_POLICY.md`; its equations,
parameterization, complexity, confirmation results, ablations, and limitations
are in `S3_NEURAL_ARCHITECTURE.md` and `S3_NEURAL_RESEARCH_REPORT.md`.

The S4 contract is `mission_s4.json`. Its no-cue boundary, derivation,
parameter formula, training protocol, ablations, per-operator results, and
failed acceptance decision are in `S4_NEURAL_PROOF_ARCHITECTURE.md` and
`S4_NEURAL_PROOF_REPORT.md`.
