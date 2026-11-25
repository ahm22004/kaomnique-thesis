# Sub-10B In/Out Multimodal Models Without Gradient Training

This repository is a thesis-only publication of the research argument and its
defense-ready claim boundary. It is not the source research repository, an
executable reproduction package, or a finished quality benchmark.

## Documents

- [Thesis](docs/thesis.md) — research question, claim, falsification criteria,
  and limitations.
- [Thesis defense](docs/thesis-defense.md) — case study, evidence boundary,
  failure analysis, licensing, and committee Q&A.

## Result at a glance

- **Method:** evidence-staged search, composition/repair, and provenance-bound
  edits without gradient-based training.
- **Kandinsky case study:** a closed-form connector is folded into two existing
  projections, producing a mixed family of **7,368,270,797** active parameters.
- **Measured result:** `text->video` produced a valid smoke artifact but received
  `answer_match: 0.0` (`n=1`).
- **Classification:** `learned_alignment`; **no target coverage claim**.

## Claim boundary

Production-geometry video quality is unmeasured. `text->audio` and
`image->video` remain explicitly unmeasured. High representation-fidelity
scores are not evidence of generation quality.

## Evidence and provenance

The path-free
[Kandinsky 6 public manifest](evidence/kandinsky6-public-manifest.json)
preserves the exact hashes, parameter census, prompt-safe split, fit
provenance, measured smoke result, and license boundary.

Raw run records, source scripts, checkpoints and weights, prompt data,
connector/state files, and generated media are intentionally excluded. This
publication is therefore a claim record, not an independently executable
reproduction.

## Related model release

The derived transformer overlay is published separately as
[ahm22004/kaomnique](https://huggingface.co/ahm22004/kaomnique).
Its combined family is restricted to non-commercial research and evaluation
under the Qwen Research License; commercial use requires a separate license
from Alibaba Cloud. That model license does not govern this thesis prose.

## Citations

- Team Kandinsky et al., *Kandinsky 6.0 Video: Foundation Models for
  Synchronized Video and Audio Generation*, arXiv:2610.05608.
- Qwen Team, *Qwen2.5-Omni Technical Report*, arXiv:2503.20215.
