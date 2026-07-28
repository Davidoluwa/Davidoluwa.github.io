# SERA S3 Neural Shape Policy

## Governing rule

SERA S3 may learn reusable parameters from anonymous input/output episodes, but
it may not receive semantic information identifying a relation. At deployment,
its global parameters are frozen before the support set is compiled into an
episode-specific shape.

## Learner-visible data

During meta-training:

1. an episode boundary;
2. raw scrambled Boolean input wires;
3. raw scrambled Boolean output wires for the support set;
4. an observation mask; and
5. complete anonymous output targets used only to calculate the meta-training
   reconstruction loss.

During confirmation:

1. a fresh episode boundary;
2. raw scrambled support input/output wires;
3. an observation mask; and
4. raw scrambled query input wires.

The confirmation outputs are never provided until after every prediction has
been frozen and recorded.

## Forbidden cues

The neural learner must not receive:

- integers or decoded numeric magnitudes;
- operand grouping;
- bit significance;
- operation identifiers, names, symbols, or descriptions;
- the input or output permutation;
- addition, subtraction, multiplication, carry, or borrow modules;
- polynomial or Hamming-weight features;
- evaluator-side relation labels;
- confirmation answers;
- evaluator feedback during confirmation; or
- per-episode gradient updates during confirmation.

Readable relation names and arithmetic implementations may exist only in the
external research harness.

## Allowed inductive bias

- The raw Boolean inputs define the vertices of a Boolean hypercube.
- Vertices differing in one raw input wire may exchange neural messages.
- The same weights are shared over all vertices, message-passing iterations,
  output wires, episodes, and evaluator-side relation families.
- Residual recurrent updates, nonlinear activations, and a supervised
  meta-training loss are permitted.

Hypercube connectivity is relation-neutral: it contains no operator identity,
operand boundary, or bit significance. It is nevertheless an architectural
prior and must not be misreported as learning without any inductive bias.

## Shape-before-answer requirement

For each episode, SERA must:

1. ingest the complete unordered support set;
2. execute all recurrent graph updates;
3. freeze the resulting whole-relation neural field;
4. stop all parameter updates; and
5. answer queries only by reading the frozen field.

It may not autoregress over answer tokens or update its parameters between
queries.

## Confirmation controls

- Every episode uses fresh input and output wire permutations.
- Confirmation seeds are disjoint from meta-training seeds.
- Support order is reversed and recompiled; output drift must be zero.
- The global parameter checksum is measured before and after confirmation.
- A checkpoint is saved, reloaded into a new process object, and checksum
  matched.
- A random-weight architecture ablation is evaluated.
- A permuted-output support control is evaluated.
- Unseen evaluator-side relation families are tested without retraining.
- Optimized and ASan/UBSan builds must produce identical prediction files.
- Every failure is retained.

## Final acceptance gate

- cue-policy violations: 0;
- support-order output drift: 0;
- confirmation parameter drift: 0;
- checkpoint checksum mismatch: 0;
- exact six-bit accuracy at 60/64 support: at least 85%;
- bit accuracy at 60/64 support: at least 96%;
- permuted-output exact accuracy: below 10%; and
- optimized and sanitized predictions: identical.

Unseen-family transfer is reported as a diagnostic rather than placed in the
gate, because its relation set was expanded after the initial architecture
choice.
