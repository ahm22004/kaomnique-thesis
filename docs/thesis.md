# Thesis: Sub-10B In/Out Multimodal Models Without Gradient Training

This document records the research argument the project is built on. It is the reason
the harness searches component combinations instead of training a model. It states what
the published evidence supports, what it does not, and what would refute the thesis.

## The question

Can a single system under **10B total parameters** take in and produce more than one
modality, run on both a legacy low-VRAM GPU and a modern GPU, and get there **without
gradient-based training** — using direct weight edits and inference/deployment tuning
instead?

## The claim

The work is feasible only as *composition and repair of already-aligned weights*, not
as *creation of alignment by editing*. Concretely:

1. A pre-aligned checkpoint supplies the cross-modal connection.
2. Search picks a combination that fits the parameter and runtime budget.
3. Edits and tuning fix diagnosed, narrow failures in that combination.
4. Any procedure that must *learn* a connection is out of the default scope and is
   reported as a different experiment class.

If a required input/output path needs an alignment no candidate checkpoint already has,
this project's default mode does not produce it. That limit is the thesis, not a
temporary gap.

## What the evidence supports

- **Weight editing updates existing behavior.** ROME/MEMIT-style locate-then-edit
  methods rewrite factual associations already represented as key–value patterns in
  MLP layers, and MEMIT scales the number of edits; EMMET adds equivalence constraints;
  AlphaEdit projects edits onto a null space to reduce collateral damage. All of these
  presuppose the model already encodes the behavior being edited.
- **Merging works inside a compatible family.** Task arithmetic and model soups combine
  checkpoints that share a pretraining lineage and are validated empirically afterward —
  they are not a mechanism for joining unrelated representations.
- **Quantization trades memory for measured quality.** Post-training quantization
  (GPTQ, AWQ, bitsandbytes, GGUF k-quants) lowers memory enough for small GPUs, with
  degradation that must be measured per model rather than assumed acceptable.
- **Alignment is learned from paired data, cheaply but not freely.** LLaVA-style
  systems train the connector against a chosen encoder and language model. Work on
  frozen unimodal encoders shows alignment can be obtained at lower cost, but it still
  uses paired data and a learned projector.
- **Runtime support is the binding constraint on legacy hardware.** Current vLLM
  requires compute capability 7.5+ (Pascal excluded) and vLLM-Omni requires 7.0+;
  llama.cpp supports broad CUDA/CPU offload with selected multimodal architectures.
  Published parameter counts of session/omni checkpoints fit under 10B, but per-model
  runner support for their submodules, preprocessing, and generation heads must be
  verified, never assumed.

## What the evidence does not support

- That tensor-dimension agreement between an encoder and a language model means the
  pair works. A projector is fitted to particular feature distributions, preprocessing,
  and token conventions.
- That a weight edit can install a cross-modal capability the checkpoint lacks.
- That a blog-level capability report constitutes a standardized edit method.
- That a benchmark measured during search establishes generalization.

## Design consequence

The search is therefore **evidence-staged**, and each stage may reject a candidate:

1. Static gates: parameter ceiling (`<10B`), declared roles, interface metadata,
   provenance, license status.
2. Runtime gates: the adapter actually loads the combination on the probed machine and
   produces finite outputs.
3. Per-path evaluation: each required input/output path is scored independently against
   its floor.
4. Pareto comparison: survivors are compared on quality and measured resource cost, with
   no aggregate allowed to hide a failed path.

Edits and tuning are searched as bounded, recorded operations applied to derived copies.
This makes every improvement attributable and keeps the no-gradient default auditable.

## How the thesis is tested

Each hypothesis has a stated failure condition, checked on data held out from candidate
selection:

| Hypothesis | Refuted if |
|---|---|
| A `<10B` combination meets the declared paths without gradient training | No combination passes interface probes and per-path floors |
| A direct edit repairs a diagnosed failure | The edit fails to improve its target path on the holdout, or breaks a passing path |
| The combination runs on the legacy target within user limits | Measured latency/memory on the probed machine exceeds the run's limits |
| Repeatability holds | A rerun with identical inputs reports different component identities or evaluation results |

A refuted hypothesis is a result to report, not a reason to relax a gate.

## Known threats

- Encoder preprocessing and token conventions differ silently between families.
- Quantization can remove exactly the precision a marginal connection depends on.
- Legacy-GPU kernel coverage changes with runtime version; record exact versions.
- Automated per-path metrics can reward shortcuts that a small human audit would reject.
- Small holdouts produce noisy floors; a pass is not proof of robustness.

## What this document does not claim

The thesis-wide candidate set is not a single finished model. The validated
combinations remain smoke-scale: they show that paths run and are scored, not
that any combination is good enough to ship. Nothing here asserts that every
requested modality pair is reachable. The source research repository's runtime
matrix and validation results record what has actually been measured; the
thesis-only publication includes a path-free public manifest but not the raw
runs, scripts, prompts, checkpoints, or generated artifacts.

The Kandinsky 6 / Qwen2.5-Omni case study is a deliberate boundary case. Its
closed-form alignment was folded into a derived transformer, the video-only
`text->video` branch executed, and the only reference-scored render was a
negative `0.0` smoke result at `n=1`. The family remains `learned_alignment` and
claims no target coverage. See [thesis-defense.md](thesis-defense.md) for the defense-ready claim
boundary, failure analysis, and anticipated questions.
