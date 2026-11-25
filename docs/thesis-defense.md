# Thesis Defense: Sub-10B In/Out Multimodal Models Without Gradient Training

Prepared from the project's measured evidence on 2026-11-25. This
document separates the thesis-wide contribution from the Kandinsky case study
so that a strong alignment result is not mistaken for a generation-quality
result.

## One-sentence answer

The work contributes an evidence-staged method for searching, executing, editing,
and auditing sub-10B multimodal component families without gradient training;
the Kandinsky 6 / Qwen2.5-Omni case study demonstrates a bounded closed-form
alignment and an executable under-ceiling path, while its `0.0` smoke score and
unmeasured production quality delimit the claim.

## Research question

> Can a system under 10B total parameters handle multiple input and output
> modalities, run under explicit resource constraints, and be improved without
> gradient-based training by composing and repairing already-aligned weights?

The original hypothesis was deliberately broader than one model. The project
therefore distinguishes:

1. **composition** of a pre-aligned family;
2. **repair** of a diagnosed failure by a bounded weight edit;
3. **creation of a new cross-modal alignment**, which requires fitting and is
   outside the default no-learning scope.

The Kandinsky work tests the third case explicitly. It does not silently promote
that case to the thesis-wide success condition.

## Thesis-wide claim and boundary

### What the thesis may claim

- A search harness can reject candidates for static, runtime, per-path, and
  resource reasons before making a coverage claim.
- Alignment provenance travels with every candidate, recipe, run, and result.
  Only `unified` and `prealigned_family` can contribute to target coverage.
- A fitted connector remains `learned_alignment`, even when its held-out
  representation fidelity is high and its weights are folded into the released
  model.
- A component family can be brought below 10B by replacing an oversized prompt
  encoder and absorbing the representation mismatch into two existing
  projections, without a gradient-training run.
- Negative and unmeasured results remain distinguishable: `0.0` is a measured
  failure; `null` with `quality_measured: false` means no measurement exists.
- Released checkpoints can be preserved byte-for-byte while all changes are
  applied to a derived copy and recorded by hash.

### What the thesis may not claim

- That the Kandinsky mixed family provides target coverage.
- That the `0.0`, `n=1` smoke result establishes production video quality.
- That `text->audio` or `image->video` has been quality-validated.
- That fitting a connector is not learning. It is closed-form parameter
  estimation, not gradient descent, but it still creates a learned alignment.
- That the mixed family is MIT-only or commercially redistributable under the
  included terms.
- That the released Kandinsky family is under 10B: its verified active total is
  12,267,462,605.
- That every requested modality pair is reachable by composition alone.

## Contributions

### 1. Evidence-staged search

Every candidate passes four independent gates:

1. static constraints: parameter ceiling, declared roles, interfaces,
   provenance, and license status;
2. runtime constraints: the selected adapter loads the candidate and produces
   finite outputs on the named device;
3. per-path evaluation: each path is scored against references and a floor;
4. comparison: survivors are compared per `(path, metric)` without allowing an
   aggregate to hide a failed path.

A rejected candidate remains useful evidence because the blocker is structured
rather than hidden.

### 2. One-alignment rule

The project uses one declared aligned architecture, not a chain of unrelated
models presented as one model. Alignment classes are recorded as:

| class | interpretation | target coverage |
|---|---|---|
| `unified` | one architecture owns the modalities | yes |
| `prealigned_family` | documented matched checkpoint family | yes |
| `external_baseline` | standalone comparison system | no |
| `learned_alignment` | parts become one architecture only after fitting | no |
| `unsupported` | no usable alignment provenance | no |

In the source research implementation, the rule is enforced in the grid,
coverage plan, run records, and Pareto output. It prevents a high score on one
path from upgrading a multi-model chain into a coverage claim.

### 3. Auditable no-gradient edits

Edits and foldings are applied only to derived copies. Every operation records
the source hash, derived hash, changed tensors, and runtime identity. The
released transformer remains byte-identical after the fold:

- released SHA-256:
  `b1b52668561d9138d93bb4da13c9999e615bf569be9f8cf65732d54652447f53`;
- derived SHA-256:
  `68243822bb2a42927fbf02ca6da9ee5a1d5dcb739966f64ee0a831f22a72bee4`.

### 4. A license-correct release boundary

The release is an overlay, not a falsely turnkey pipeline. It separates the
distributed derived transformer from the separately downloaded Kandinsky
components and Qwen GGUF, includes the Qwen Research License and MIT terms, and
marks the modified tensors and config.

## Kandinsky case study: architecture

### Released family

Kandinsky 6.0 Video Lite is a dual-stream text-to-audio-video diffusion model.
The released Diffusers family is:

| component | active parameters |
|---|---:|
| transformer | 3,177,988,112 |
| Qwen2.5-VL-7B prompt encoder | 8,292,166,656 |
| CLIP pooled encoder | 123,060,480 |
| video VAE | 246,478,803 |
| audio VAE | 305,613,186 |
| vocoder | 122,155,368 |
| **total** | **12,267,462,605** |

The `audio_vae` file serializes a duplicate copy of the vocoder. It is excluded
from the active count and counted once through the separate vocoder component.
The released family is therefore above the 10B ceiling by 2,267,462,605
parameters.

### Mixed family

The released 8.29B Qwen2.5-VL prompt encoder is replaced by the 3,397,103,616
parameter Qwen2.5-Omni-3B GGUF. Folding the two text projections from 3,584 to
2,048 inputs also removes 4,128,768 parameters from the active transformer. The
project accounting total is **7,368,270,797**, below the ceiling.

The two encoders do not share a coordinate frame. Kandinsky expects token states
from Qwen2.5-VL-7B:

- video token projection: `[1792, 3584]`;
- audio token projection: `[896, 3584]`.

Qwen2.5-Omni supplies 2048-dimensional token states. A linear connector
`A: 2048 -> 3584` can therefore be folded into the two existing matrices:

```text
W_new_video = W_released_video @ A^T
W_new_audio = W_released_audio @ A^T
```

After folding:

- video projection: `[1792, 2048]`;
- audio projection: `[896, 2048]`;
- `in_text_dim`: `2048`.

There is no runtime connector and no added transformer block. The CLIP pooled
path is unchanged. Image conditioning uses the released video VAE and adds no
extra encoder.

## Data split and method selection

### Fixed prompt split

The alignment prompts were fixed before fitting:

| split | prompts | tokens | use |
|---|---:|---:|---|
| training pool | 60 | 1,658 | candidate fitting and validation selection |
| training-pool validation | 4 | 153 | choose state-space or branch-space method |
| holdout | 18 | 453 | score only the selected method |

The validation prompts are drawn from the earlier training pool, so the
selection does not consume holdout information. Both selection candidates are
fitted on the earlier 1,505 tokens. After the method is selected, its state-space
connector is refitted on all 1,658 training tokens. Only that final connector is
evaluated on the holdout; unselected candidates' holdout scores are not reported.

### Candidates

1. **State-space candidate:** regress Qwen2.5-VL hidden states from Qwen2.5-Omni
   hidden states, then fold into the released projections.
2. **Branch-space candidate:** regress the released branch's own embedding
   directly from Qwen2.5-Omni states.

Both are closed-form ridge regressions. No optimizer, minibatch, backpropagation,
or gradient checkpoint is used.

### Selection result

| branch | state validation cosine | branch validation cosine | selected |
|---|---:|---:|---|
| video | **0.989182** | 0.981849 | state |
| audio | **0.972654** | 0.947324 | state |

On the holdout, the selected state-space connector refitted on all 1,658
training tokens yields:

| branch | cosine | R² |
|---|---:|---:|
| video | **0.993369** | **0.910443** |
| audio | **0.985168** | **0.845637** |

A fixed-coordinate slice control fails (holdout cosine `-0.004`), showing that
the two encoders do not already share coordinates and that the fit is doing
real work.

### Interpretation

The high fold scores mean that the transformer receives nearly the same text
branch embedding as the measured Qwen2.5-VL target would produce on these
prompts. They do **not** mean that the generated pixels or audio are correct.
The fit reduces one diagnosed failure (representation mismatch) while leaving
semantic generation quality as a separate question.

## End-to-end evidence

### Component verification

The component manifest verifies the released model files against published
SHA-256 values and sizes, counts exact safetensors parameters from headers,
checks required tensors for all-zero content, and excludes the duplicated nested
vocoder. The verified result has no errors and no required zero tensors.

### Smoke render

| field | value |
|---|---|
| path | `text->video` |
| geometry | 16×16, 5 frames, 24 fps, 1 inference step, seed 0 |
| device | CPU, BF16 |
| runtime | PyTorch `2.15.0+cpu`, Diffusers `0.43.0`, Transformers `5.23.0` |
| model load | 3,751.3 ms |
| generation | 89,203.6 ms |
| total wall time | 92,955.0 ms |
| runner-process peak memory | 5,150.8 MB |
| artifact | valid five-frame H.264 MP4, 1,775 bytes |
| judge | local Qwen2.5-Omni video judge answered “No.” |
| score | `answer_match: 0.0`, `n=1`, `quality_measured: true` |

This proves that component verification, Qwen token conditioning, CLIP pooling,
the folded transformer, video VAE, MP4 writing, the local video judge, and the
run-record pipeline execute together. The audio VAE and vocoder were hash-verified
but were not loaded or executed because the smoke requested video only. It is a
real negative result.

### Why the smoke was deliberately tiny

The available machine has a GTX 980 with compute capability 5.2 and an installed
CPU-only PyTorch build. The checkpoint's BF16 CUDA kernels require a newer
architecture. A CPU 16×16, five-frame, one-step render took roughly 89–216
seconds across runs. Production geometry and a meaningful multi-step reference
set require different hardware, currently a free GPU run that has not been
performed.

## Failures and limitations

1. **Quality failure:** the only reference-scored render failed the reference
   check (`0.0`, `n=1`).
2. **No production benchmark:** geometry, step count, and sample count are too
   small for a quality or coverage claim.
3. **Two paths unmeasured:** `text->audio` and `image->video` have implementation
   scoping but no live reference-scored evidence.
4. **Small alignment sample:** 18 holdout prompts and 453 tokens characterize
   the measured representation, not every possible prompt.
5. **Quantized target:** the measured target states were produced from the
   published Qwen2.5-VL-7B Q4_K_M build and therefore include that checkpoint's
   quantization error. The included provenance does not establish the exact
   precision used for training.
6. **One replaced path only:** CLIP pooled conditioning remains exactly as
   released; only token conditioning is replaced.
7. **Fidelity is not quality:** matching the released branch embedding does not
   guarantee a faithful render.
8. **Legacy-GPU gap:** Qwen2.5-Omni reading paths run through llama.cpp's
   Vulkan backend on the GTX 980, but the BF16 Kandinsky diffusion path cannot
   use that GPU. One model family across all modalities on the legacy device is
   therefore not demonstrated.
9. **License restriction:** commercial use of the mixed family is not permitted
   by the included Qwen Research License.

## Licensing defense

The license answer must name both sources:

- Kandinsky 6.0 Video Lite source code, checkpoints, and Diffusers integration
  are MIT-licensed.
- Qwen2.5-Omni is under the Qwen Research License: non-commercial research and
  evaluation only. Commercial use requires a separate license from Alibaba
  Cloud.
- The Qwen2.5-VL-7B alignment target is Apache-2.0, but its weights are not
  distributed by the mixed-family release.

Therefore the combined family is `license: other`, with license name
`qwen-research`. It is not MIT-only. The release includes the Qwen agreement, the
MIT terms, the exact required notice, “Improved using Qwen,” and explicit notices
for the modified tensors and config.

## Anticipated committee questions

### 1. You fitted a connector. How is that “without gradient training”?

The connector parameters were estimated by closed-form ridge regression, not by
gradient descent. The thesis distinguishes **no gradient-based training** from
**no learned parameters**. This case does create a learned alignment, so it is
classified `learned_alignment` and excluded from target coverage.

### 2. If the folded cosine is 0.993, why is the family not aligned?

Because cosine measures only the selected token branch on 18 held-out prompts.
It does not measure semantic correctness, temporal consistency, audio quality,
or generation success. The classification follows the kind of operation required
to make the parts one architecture, not the attractiveness of one intermediate
metric.

### 3. The measured score is 0.0. Why release the model?

The release preserves a verified construction and a negative result. Its value
is reproducibility and diagnosis: a user can verify the exact source, fold, and
payload rather than inherit an unfalsifiable “it works” claim. The card and
manifest state the failure prominently and make no production-quality claim.

### 4. Why not release the complete mixed pipeline?

The complete mixed family does not depend on the released 8.29B encoder. It depends on the separate
2.1GB Qwen2.5-Omni GGUF prompt encoder, multiple large codec and CLIP files, and the derived
transformer. Publishing that derived transformer plus an exact dependency manifest avoids an
ambiguous or incomplete repository and preserves the component boundaries used by the
measurements.

### 5. Why not count only the 3.17B derived generation transformer?

That would change the meaning of “under 10B” for every family and could hide a
large prompt encoder. The accepted architecture decision counts every active
part of the family once. The release follows that rule and reports the released
family as 12.27B.

### 6. Why is the mixed total 7,368,270,797?

That is the project's accepted active-family accounting after replacing the
8.292B prompt encoder with the measured 3.397B Qwen2.5-Omni encoder, subtracting
the 4,128,768 parameters removed by folding both text projections, and counting
the nested vocoder once. The count comes from checkpoint metadata and published
index evidence, not a model-card estimate.

### 7. Why not count the connector as 7.34M additional runtime parameters?

The connector is folded into existing projection matrices and is not present at
runtime. The fold therefore adds no connector module or runtime parameter
tensor. The connector and its evidence hashes remain in the audit trail.

### 8. How reliable is the judge?

The local Qwen video judge is not treated as infallible. Its calibration and
negative control show that it answers both yes and no on controlled clips. The
`n=1` sample still makes the result a smoke result. A future benchmark needs
more samples, production geometry, judge calibration on that set, and a negative
control.

### 9. Why was the render so small?

The local BF16 diffusion path cannot execute on the sm_52 GTX 980, and the CPU
path takes minutes even at smoke geometry. Tiny geometry was the only honest
local smoke test. It was explicitly not labeled a benchmark.

### 10. What is reproducible?

The component hashes, parameter census, split, connector fit, method selection,
folded tensor hashes, derived payload hash, runtime versions, geometry, and
scored artifact were recorded in the source research repository. The public
manifest distributes the path-free subset. The source fold script refuses to
overwrite the released source and verifies byte-level tensor layout independently
of its own arithmetic.

### 11. What is the strongest contribution if the target frontier did not expand?

The contribution is not one lucky checkpoint. It is the evidence architecture:
static gates, runtime gates, reference scoring, prompt-safe selection,
provenance-bound edits, exact failure reporting, and a licensing boundary that
prevents a technically executable family from being mislabeled as a successful
target system.

### 12. What is the next experiment?

On free GPU hardware:

1. run production geometry and a multi-step schedule;
2. score at least a pre-registered prompt set with the same local judge and a
   negative control;
3. evaluate `text->video`, `text->audio`, and `image->video` independently;
4. repeat with seeds to measure stability;
5. compare against the released 12.27B family on the same references;
6. update coverage only if the pre-registered floors pass.

The family would remain `learned_alignment` regardless of the score unless the
project separately changes its architecture-class rule.

## Suggested defense sequence (12 slides)

1. **Question and thesis:** what “without gradient training” can and cannot mean.
2. **Contribution:** evidence-staged search, one-alignment rule, auditable edits.
3. **Why Kandinsky 6:** one family, multiple generation paths, but 12.27B.
4. **Exact census:** released components, nested-vocoder correction, ceiling.
5. **Architecture change:** replace 8.29B encoder with 3.397B Qwen2.5-Omni.
6. **Fold:** two matrices, equation, no runtime connector.
7. **Leakage-safe method:** fit, validation selection, one holdout score.
8. **Representation evidence:** 0.993 video / 0.985 audio cosine.
9. **End-to-end evidence:** tiny render, `0.0`, `n=1`, valid artifact.
10. **Failure analysis:** what the negative result does and does not show.
11. **License and release:** MIT + Qwen Research, non-commercial, overlay scope.
12. **Next experiment and claim boundary:** what evidence would change the thesis.

## Defense checklist

Before the defense:

- [ ] Re-run the source research component verifier and confirm `verified: true`
      with no errors.
- [ ] Re-check the released and derived transformer SHA-256 values.
- [ ] Confirm the public manifest contains no local paths.
- [ ] Confirm the public model card still says `0.0`, `n=1`, smoke-only,
      `learned_alignment`, and that audio/image-to-video remain unmeasured.
- [ ] Show the source-unchanged proof and the two folded tensor hashes.
- [ ] Re-run the local judge calibration and negative control if a live clip is
      used in the demonstration.
- [ ] Have a fallback: the recorded run record, path-free manifest, and inline
      architecture summary in this document; do not depend on a live GPU render.
- [ ] Keep commercial-use questions explicit: the included Qwen Research
      License permits research/evaluation only.

## Closing formulation

A defensible closing statement is:

> We built and tested a falsifiable route to a sub-10B multimodal generation
> family without gradient training: replace an oversized prompt encoder, fit a
> bounded closed-form alignment on a prompt-safe split, fold it into existing
> projections, and verify every artifact by hash. The representation fidelity is
> high and the path executes, but the only end-to-end reference score is a
> `0.0` smoke failure at `n=1`; production quality and two generation paths
> remain unmeasured. The result is therefore evidence of an auditable method and
> executable construction, not a completed quality claim.
