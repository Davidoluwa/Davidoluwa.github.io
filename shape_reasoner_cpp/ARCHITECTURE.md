# SERA: Shape Equilibrium Reasoning Architecture

## Research claim

Reasoning should not be identified with producing a sequence of words. SERA
treats reasoning as the formation and resolution of a structured latent object:
a **shape**. A shape holds entities, variable bindings, relations, goals,
uncertainty, competing hypotheses, and transformations simultaneously.

Language is downstream:

```text
observations -> shape formation -> equilibrium / topology edits
             -> semantic skeleton -> optional language rendering
```

S0 tests the continuous equilibrium substrate. S1 adds a discrete categorical
kernel, active-frontier settlement, and adaptive shape particles. Neither stage
claims to solve grounding, representation discovery, or natural-language
generation.

## Why "shape" is more than a vector

A single dense vector erases the distinction between:

- an entity and one of its properties;
- two entities that share a property;
- evidence and a goal;
- a causal relation and a correlation;
- one hypothesis and a mutually exclusive alternative.

SERA therefore defines a shape as

\[
\mathcal S=(V,F,X,Q,\Pi),
\]

where:

- \(V\) is a set of typed identity-bearing slots;
- \(X=\{x_i\}\) are continuous local states;
- \(F\) is a sparse set of typed relational factors;
- \(Q\) is a goal/query field that marks what must be resolved;
- \(\Pi\) is a set of competing shape particles when the topology or answer is
  ambiguous.

The topology is part of the representation. Two shapes can contain similar
local values while expressing different propositions because their bindings and
relations differ.

## Whole-shape reasoning

Every factor \(f\) defines a local incompatibility energy
\(\phi_f(X_f;\theta_f)\). Evidence and goal terms clamp or bias parts of the
shape. The settled interpretation is

\[
X^\*=\arg\min_X
\left[
\sum_{f\in F} w_f\,\rho\!\left(\phi_f(X_f;\theta_f)\right)
+ \sum_i \lambda_i\|x_i-e_i\|^2
+ \Omega(X,\mathcal S)
\right].
\]

No word is selected during this process. All active relations contribute to the
same global state. In the linear S0 kernel, the energy is a sparse weighted
least-squares system solved by matrix-free preconditioned conjugate gradients.
S1 implements general pairwise categorical costs with damped min-plus
settlement and particle competition. Topology edits remain future work.

### Error-gated computation

Factors whose residual remains below a calibrated tolerance become dormant.
Unexpected evidence, a changed goal, or a counterfactual reactivates only the
affected causal neighborhood. This carries forward the prior result that useful
compute should scale with surprise and replanning rather than total stored
knowledge.

The exact linear solver in S0 evaluates the fixed factor set so its mathematical
solution stays auditable. S1 implements a deterministic causal active frontier
and reconciles its candidate-operation ledger against dense-sweep work.

### Shape particles

When one topology cannot represent the uncertainty, SERA maintains a small set
of particles \(\Pi=\{\mathcal S_k\}\). Particles settle in parallel, and their
scores combine constraint energy, evidence coverage, complexity, and predictive
accuracy:

\[
\mathcal F_k=E(\mathcal S_k)+
\beta\,C(\mathcal S_k)-
\gamma\,\mathrm{coverage}(\mathcal S_k).
\]

Particles are not word continuations. They are alternative structured
interpretations that may share subshapes.

## Topology formation and revision

SERA separates two operations that token models often blur:

1. **State inference:** settle values inside a fixed topology.
2. **Structure inference:** add, remove, merge, split, or retype nodes/factors
   when residuals cannot be explained by state changes.

Topology edits are proposed only when telemetry shows a reason:

- energy plateaus above tolerance;
- residuals cluster around a missing relation;
- two slots repeatedly co-vary and should bind;
- one slot exhibits incompatible dynamics and should split;
- an existing factor predicts poorly across episodes.

This is the architecture's metacognitive boundary. A topology edit must reduce
held-out predictive energy enough to pay for its complexity and compute.

## Learning without global token backpropagation

The intended learning hierarchy is:

1. local factor parameters learn from their own prediction residuals;
2. factor templates consolidate repeated local transition shapes;
3. the selector learns which factors belong in a goal-conditioned subshape;
4. topology operators are reinforced by evidence reduction per unit compute;
5. stable settled shapes enter episodic and semantic memory.

The S0 engine contains no learned parameters. This is deliberate: solver
correctness and telemetry must be established before learning is allowed to
hide substrate failures.

## Semantic skeleton compiler

After settlement, a deterministic compiler extracts a compact intermediate
representation:

- resolved entities and variable bindings;
- high-confidence relations;
- remaining contradictions and uncertainty;
- causal dependencies supporting the answer;
- selected plan/operator outline;
- provenance back to evidence factors.

The skeleton can be serialized as compact binary or JSON. A separate renderer
may translate it into language. Rendering errors cannot feed back and rewrite
the solved shape unless explicitly entered as new evidence.

## CPU-first implementation rules

- C++20 core with a narrow C-compatible boundary planned for embedding.
- Structure-of-arrays state and contiguous factor terms.
- Matrix-free sparse operators; no dense \(N\times N\) matrices.
- Preallocated solve workspace; no allocations inside an iteration.
- Deterministic seeds and stable factor identifiers.
- Goal-conditioned connected-component extraction before expensive inference.
- ROM/mmap-friendly factor templates and memories in later stages.
- Telemetry is a first-class data product, not debug print statements.

## Required telemetry

Every solve records:

- initial/final energy;
- residual norm and convergence slope;
- iterations, matrix-vector products, and factor evaluations;
- active constraint fraction;
- maximum state update;
- wall-clock time;
- graph size, dimensionality, and goal-slice size;
- contradiction score and high-residual factor identifiers;
- deterministic hashes for graph and output.

Later learned stages add selector precision/recall, topology edit utility,
particle diversity, memory retrieval entropy, and compute per resolved
uncertainty.

## Adjacent work and non-novel components

SERA is not claiming invention of latent reasoning, factor graphs, equilibrium
models, object slots, vector binding, or graph reasoning. Relevant neighboring
ideas include:

- continuous latent reasoning such as Coconut;
- object-centric slots and object-level latent prediction;
- energy-based models and equilibrium propagation;
- deep equilibrium fixed-point models;
- graph neural algorithmic reasoning;
- tensor-product and vector-symbolic binding.

The research question is whether these ingredients can be reorganized into a
CPU-first system where:

1. the topology itself is the latent thought;
2. goals select a causal subshape;
3. global settlement precedes communication;
4. local residuals drive learning and structural revision;
5. output language is only a compiled view.

That combination must earn its value experimentally.

## Stage roadmap

### S0 — Auditable equilibrium substrate

Linear relational factors, exact matrix-free solver, goal slicing, semantic
skeleton, deterministic telemetry, contradiction stress.

### S1 — Categorical shapes (implemented)

General pairwise categorical factors, counterfactual min-plus messages,
deterministic active-frontier settlement, recursive correlated particles,
exact-oracle comparison, convergence and malformed-input controls.

### S2 — Learned relevance

Replace deterministic goal slicing with a learned selector. Test branching,
merging, adversarial distractors, held-out sizes, and selector overhead.

### S3 — Learned topology

Slot creation, identity persistence, merge/split operations, and factor-template
learning from transitions.

### S4 — Grounding and communication

Ground shapes from nonlinguistic environments first; then add a language parser
and skeleton renderer whose failures are measured independently.

## Falsification conditions

The architecture should be abandoned or materially revised if:

- useful shapes cannot be learned without hand-coded task topology;
- equilibrium cost grows as badly as token-by-token search;
- topology revision produces uncontrolled rule/factor explosion;
- a dense baseline matches results at lower end-to-end compute;
- learned selectors fail under adversarially similar distractors;
- the semantic skeleton loses information needed for reliable communication.
