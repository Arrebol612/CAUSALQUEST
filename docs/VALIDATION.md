# Validation

Date: 2026-11-07. The public package passed **38 local tests**. A separate installed-wheel smoke
checks import isolation and the complete toy SFT/review/RL/reload workflow. The GitHub CPU workflow
runs these package tests and the demo on pushes and pull requests.

| Environment | Versions |
|---|---|
| Local CPU tests | Python 3.10.20, Torch 2.11.0+cu128, Transformers 4.55.2, PEFT 0.18.1, Accelerate 1.10.1, DuckDB 1.5.6 |
| Real single-A40 acceptance | Python 3.11, Torch 2.8.0+cu128, Transformers 4.55.2, PEFT 0.18.1, Accelerate 1.10.1, DuckDB 1.5.5 |

## Real Qwen3-4B/A40 workflow

One SFT update, grouped autonomous task sampling, full-cohort public-evidence review and one RL update
completed with the existing recovery-SFT R60 adapter as the starting policy and fixed reference.
The cases are three authored synthetic TRAIN fixtures. The R60 weights are supplied separately;
this repository contains no original model weights or private research data.

| Stage or check | Measured result |
|---|---|
| SFT | 13 assistant actions, 1 update, fresh reload verified |
| Actual gradient-checkpoint execution during SFT | 468 calls |
| SFT peak allocated VRAM | 16194.9 MiB |
| Final sampling cohort | 3 cases × 2 seeds, 6 complete observed episodes, 91 model responses |
| Sampling profile | Temperature 0.6, top-p 1, top-k 0; model-default fallback disabled |
| Pre-update human review | 2/6 submitted, 1/6 core supported, 0/6 strict complete |
| Supported policy-advantage groups | 1/3; other groups retained with reference regularization |
| RL batch | All 6 trajectories, 16318 target tokens, 1 optimizer update |
| Pre-update current/old HF maximum log-probability difference | 0.0 |
| Actual Qwen parameter gradient norm before clipping | 0.2699314 |
| Changed parameters and fresh reload | Verified |
| Fixed R60 reference and original model files | Preserved |
| RL peak allocated VRAM | 24028.8 MiB, approximately 23.5 GiB |
| RL job elapsed time | 12 minutes 33 seconds |
| Post-update sampled k3 to fixed reference | 0.0004285 |
| Post-update clip fraction versus old | 0.0037995, approximately 0.38% |

All **223 files** in the update archive were checked against its export manifest. The full run,
all 91 generation requests/completions, episode files, review bindings and new checkpoint identity
agree. Saved-weight bytes remain on the server; native save/reload records certify their verification.
The sampled k3 and clipping diagnostics use the sampled training tokens. Independent post-update
quality evaluation remains future research. The small, repeatedly used integration panel provides no
training-benefit or generalization comparison. See [the portable result record](acceptance_v0.1.json).

## What the software tests cover

- Real Torch updates and the learned toy policy: SFT, complete grouped sampling, review import, one
  ordinal RL update, changed parameters, fresh reload and fixed-denominator evaluation.
- Locally initialized tiny GPT2 and Qwen3 models with real PEFT LoRA: causal completion log probabilities,
  assistant masks, frozen-base retention and uninterrupted-versus-resumed Adam equivalence.
- Real generation logits processors under model-specific defaults: requested temperatures 1 and 0.6,
  top-p 1 and top-k 0 reach the sampler; the tokenizer's zero-valued pad ID remains valid.
- Actual decoder checkpoint execution, matching checkpointed/uncheckpointed gradients, and eval-mode
  generation. A separate small-Qwen3 activation-memory check also confirmed the correction.
- Domain tools and boundaries: three task families, all five numeric helpers, read-only observations,
  prior-evidence references, state/report phases, hidden-document rejection and the shared call budget.
- Task Rubric Tree JSON examples match all three public builders; nested local weights, duplicate
  node identifiers and invalid sibling totals are checked.
- The standalone domain tutorial builds all three case families and replays its 13 authored actions
  without a model, paid API, private experiment directory or archive records.
- Finite differences, detached old/behavior/reference probabilities, advantage clipping, reference
  gradients in flat groups and reduction over unequal-length trajectories.
- Pending/duplicated/mismatched reviews, invalid evidence, stale actors, non-TRAIN updates, split/source
  leakage, nonfinite JSON, duplicate keys, exclusive writes and repeated-output refusal.

Site-only checks additionally replayed all 13 original SFT token sequences, three fresh environment
prompts and the native generation receipts with the pinned Qwen3 tokenizer. The zero-signal cohort
blocks RL submission before model loading. The final reviewed cohort's native probability-coordinate
policy derivative passed over all 16318 tokens; actual Qwen gradients were then verified on the A40.
Site-only scripts and paths remain outside the public package.

## Corrections discovered during acceptance

The first A40 SFT attempt exhausted VRAM because the decoder remained in eval mode, bypassing its
gradient-checkpoint path. Grad-enabled scoring now enables decoder training while disabling stochastic
dropout and checks actual checkpoint calls. The corrected A40 SFT run completed within the measured
memory budget.

The first sampling cohort exposed generation-default fallback and live-versus-persisted tool-schema
ordering differences. Canonical schema rendering and explicit fallback control now keep SFT, live
requests and stored token records consistent. Earlier cohorts remain separate diagnostic evidence;
the RL update used only the final temperature-0.6/top-p-1 cohort. Model quality remains weak: a real
missing-observation finding supports the health action, while false-positive slice diagnoses, design
repair errors and report-schema failures remain visible in the scores.
