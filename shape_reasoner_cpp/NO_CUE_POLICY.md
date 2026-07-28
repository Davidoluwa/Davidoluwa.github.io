# SERA Strict No-Cue Learning Policy

## Governing rule

The learner may infer a hidden relation only from demonstrations expressed in
the same raw representation as its queries. It may not receive semantic,
structural, evaluator-derived, or test-derived information that identifies the
relation or reveals the intended answer.

## Permitted input

1. An episode boundary stating only that a set of demonstrations shares one
   hidden mapping.
2. Fixed-width Boolean input vectors.
3. Fixed-width Boolean output vectors paired with training inputs.
4. Unlabeled Boolean query vectors after the model is frozen.

The episode boundary is unavoidable: without knowing which demonstrations
belong to one mapping, a mixture of mutually inconsistent mappings is not an
identifiable learning problem.

## Forbidden cues

The learner must not receive:

- integer values or numeric decoding;
- operand boundaries or grouping;
- bit significance or positional weights;
- operator identifiers, names, symbols, descriptions, or examples tagged by
  operation;
- an arithmetic grammar;
- addition, subtraction, multiplication, division, carry, or borrow
  primitives;
- polynomial features;
- output-wire significance;
- test outputs;
- evaluator feedback after inference;
- a development-set score during the final confirmation run;
- a hand-selected program or formula for any episode.

## Allowed inductive bias

The learner may use a relation-neutral, functionally complete Boolean
representation. The final system uses algebraic normal form:

\[
f(x)=\bigoplus_{S\subseteq\{1,\ldots,n\}}
c_S\bigwedge_{i\in S}x_i.
\]

This is not an arithmetic cue: every Boolean function has exactly one complete
ANF representation.

Under partial demonstrations, the consistent coefficient vectors form an
affine space over \(\mathrm{GF}(2)\). SERA may search that version space and
choose the minimum-description consistent program.

## Enforcement

- All six input wires are randomly permuted independently for every episode.
- All six output wires are randomly permuted independently for every episode.
- The learner receives neither permutation.
- Training and withheld minterms are disjoint and checked at runtime.
- The relation evaluator exists only in the research harness.
- The generic learner is implemented in a separate source file with no oracle,
  operator symbol, or arithmetic dispatch.
- Models are frozen before withheld evaluation.
- Training example order is shuffled and retrained; output drift must be zero.
- A negative control permutes output demonstrations before executing the
  identical training and evaluation pipeline.
- Fresh confirmation episodes use seeds and wire permutations not used during
  architecture development.
- Every prediction, including failures, is retained.

## Final acceptance gate

- cue-policy violations: 0;
- train/test overlap: 0;
- training errors: 0;
- training-order output drift: 0;
- overall exact-word accuracy: at least 80%;
- overall bit accuracy: at least 95%;
- permuted-output exact-word accuracy: below 10%;
- release build and ASan/UBSan build must produce identical aggregate results.

Passing this gate demonstrates cue-free relation induction within the bounded
six-bit Boolean environment. It is not evidence of unrestricted arithmetic or
general intelligence.

