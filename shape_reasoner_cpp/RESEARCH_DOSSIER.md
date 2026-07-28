# SERA Research Dossier

## Thesis

The architecture should reason in a structured latent medium and communicate
only after that internal structure has settled. The internal medium is not a
sentence and not one undifferentiated vector. It is a sparse relational shape
that can represent identity, bindings, evidence, goals, uncertainty,
counterfactual alternatives, and causal support simultaneously.

This is a hypothesis about useful machine computation. It does not depend on
the stronger and scientifically unsettled claim that humans always formulate a
complete nonverbal answer before speaking. Human studies do demonstrate
nonverbal deduction and abstract task-variable geometry, which makes the design
direction plausible rather than proven.

## Breadth map

| Neighboring field | What it already contributes | What remains different in SERA |
|---|---|---|
| Continuous latent reasoning | Reasoning need not decode every intermediate state into words; Coconut reports latent states that can carry multiple alternatives. | SERA's thought is an explicit variable-sized relational topology, not a hidden-state trajectory recycled through a language model. |
| Deep equilibrium models | Fixed-point computation can replace explicit depth and use constant activation memory. | SERA exposes typed factors, evidence, contradictions, goals, and provenance rather than treating the equilibrium as an opaque layer. |
| Energy-based models / equilibrium propagation | Global compatibility can be expressed as energy; local perturbations can support learning. | SERA separates state settlement from topology revision and requires compute/uncertainty telemetry for each structural edit. |
| Object-centric learning | Exchangeable slots can represent entities and improve compositional structure. | SERA requires persistent identity, typed relations, causal support and goal-conditioned subshape selection beyond object discovery. |
| Graph neural algorithmic reasoning | Graph structure is a strong substrate for algorithmic and combinatorial reasoning. | SERA does not require a fixed message-passing depth or imitation of a known step sequence; it settles a factor field and later permits topology particles. |
| Tensor products / vector-symbolic architectures | Role-filler binding and algebraic composition can encode structured objects in distributed vectors. | SERA can use binding inside slots or memory signatures, but topology remains explicit so collisions do not silently erase causal structure. |
| JEPA-style latent world models | Predictive embeddings can discard irrelevant observation detail; object-level masking can induce interaction reasoning. | SERA targets a general reasoning state with explicit goals, contradictions, provenance and counterfactual topology, not only latent prediction. |
| Classical factor graphs and sparse solvers | Constraint systems can be represented and solved globally with strong mathematical guarantees. | S0 intentionally inherits this substrate. Novel value, if any, must come from learned factor formation, goal selection, topology revision, memory and communication. |

## Closest current approaches

### Coconut

[Training Large Language Models to Reason in a Continuous Latent
Space](https://arxiv.org/abs/2412.06769) feeds an LLM hidden state back as a
continuous thought rather than decoding a word. This is important evidence
against treating text as the only reasoning medium. It remains a sequential
latent-state process inside a pretrained language architecture.

### Deep equilibrium and equilibrium propagation

[Deep Equilibrium Models](https://arxiv.org/abs/1909.01377) solve for a
fixed-point hidden state rather than materializing a conventional deep stack.
[Equilibrium Propagation](https://arxiv.org/abs/1602.05179) provides a
well-defined energy-based learning framework using free and nudged phases.
SERA borrows the equilibrium viewpoint, while requiring factor-level
interpretability and explicit topology.

### Object-centric structure

[Slot Attention](https://arxiv.org/abs/2006.15055) produces exchangeable
object-centric slots through iterative competition. [Causal-JEPA](https://arxiv.org/abs/2602.11389)
uses object-level masking to make interaction-dependent prediction necessary.
These are relevant candidates for future shape formation, but neither alone
supplies SERA's reasoning, goal, memory or topology-revision system.

### Graph algorithmic reasoning

[Graph Neural Networks are Dynamic Programmers](https://arxiv.org/abs/2203.15544)
formalizes links between GNN computation and dynamic programming.
[Deep Equilibrium Algorithmic Reasoning](https://arxiv.org/abs/2410.15059)
brings fixed-point computation into neural algorithmic reasoning. SERA must
demonstrate benefits beyond these graph/equilibrium combinations, especially
when topology is not supplied.

### Compositional representations

[Tensor Product Variable Binding](https://www.lscp.net/persons/dupoux/teaching/AT1_2014/papers/Smolensky_1990_TensorProductVariableBinding.AI.pdf)
and modern vector-symbolic architectures provide algebra for binding roles and
fillers. These may solve the memory-signature problem left open by the earlier
low-entropy symbol-set index, but must be stress-tested for interference and
cleanup cost.

## Mathematical core

For a fixed shape,

\[
E(X;\mathcal S)=\sum_f w_f\rho(\phi_f(X_f))
+E_{\mathrm{evidence}}+E_{\mathrm{goal}}+\Omega(X).
\]

The resolved state is a stationary point or distribution over stationary
points. A useful solver must provide more than an answer:

\[
(X^\*,\, E^\*,\, R_f,\, U,\, C),
\]

where \(R_f\) are factor residuals, \(U\) is calibrated uncertainty, and \(C\)
is compute consumed. The research objective is not minimum task loss alone:

\[
\mathcal J =
L_{\mathrm{task}}
+\lambda_E E^\*
+\lambda_C C
+\lambda_K K(\mathcal S)
+\lambda_U L_{\mathrm{calibration}}.
\]

Topology proposals \(T\) should be accepted only when their held-out reduction
in unexplained energy exceeds complexity and compute:

\[
\Delta(T)=
E_{\mathrm{before}}-E_{\mathrm{after}}
-\alpha\,\Delta K
-\beta\,\Delta C > 0.
\]

This prevents factor/rule explosion from masquerading as intelligence.

## The actual proposed architectural contribution

No individual component is new. The candidate contribution is the enforced
division of labor:

1. **Grounder:** observations become provisional slots and evidence factors.
2. **Shape former:** binding and relation hypotheses create several sparse
   topology particles.
3. **Goal field:** the task changes activation and subshape selection without
   rewriting stored knowledge.
4. **Equilibrium core:** active particles settle globally.
5. **Residual critic:** unexplained structure triggers only justified topology
   edits or memory retrieval.
6. **Consolidator:** recurring local settled subshapes become reusable factor
   templates.
7. **Skeleton compiler:** a deterministic high-level representation preserves
   answer, uncertainty, support and contradictions.
8. **Renderer:** language or action is generated from the skeleton and is
   evaluated separately.

The architecture fails if learned grounding cannot produce useful topologies,
or if the cost of shape construction and settlement erases the savings from
not reasoning through tokens.

## Research axes

### Representation

- Can stable entities emerge without segmentation labels?
- Can identity survive occlusion, renaming and viewpoint change?
- Do role-filler bindings preserve branching causal structure?
- Does the topology remain compact as knowledge grows?

### Reasoning

- Does global settlement outperform sequential commitment under ambiguity?
- Can particles represent alternatives without combinatorial explosion?
- Can nonlinear local factors compose out of distribution?
- Does a settled shape support counterfactual edits cheaply?

### Learning

- Can factor templates learn through local predictive residuals?
- Does topology learning require global backpropagation?
- Can error-gated consolidation separate forming from stable knowledge?
- Can one-shot episodes alter memory without corrupting semantic factors?

### Compute

- End-to-end cycles per reduced uncertainty.
- Cache misses and bytes moved per active factor.
- Selector overhead versus factors skipped.
- Scaling with stored knowledge versus active causal neighborhood.
- Worst-case convergence and particle count.

### Communication

- Skeleton sufficiency for faithful verbalization.
- Renderer hallucination rate conditioned on a correct skeleton.
- Whether language feedback improves grounding or contaminates reasoning.
- Cross-lingual rendering from the same settled shape.

## Immediate experimental sequence

1. **S1 nonlinear ambiguity:** categorical beliefs, mutually exclusive
   hypotheses and particle competition.
2. **S2 learned relevance:** replace deterministic connected-component slicing
   with a selector; use adversarially similar distractors.
3. **S3 topology discovery:** infer nodes and relations from state transitions,
   beginning with nonlinguistic grid/video environments.
4. **S4 local learning:** consolidate factor templates through residual-driven
   updates and compare against global backpropagation.
5. **S5 skeleton communication:** add a small renderer only after skeleton
   sufficiency is measured.

Each stage requires a pre-registered baseline and failure boundary. No language
demo should be treated as evidence of reasoning.
