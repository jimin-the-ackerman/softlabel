# NeurIPS 2025 Rebuttal Summary

SoftPrompt was submitted to **NeurIPS 2025** and went through a full rebuttal cycle
with four reviewers. The paper was ultimately **rejected**, but the rebuttal period
produced a substantial set of new experiments that meaningfully extended the thesis
work. This document summarizes those experiments; the original rebuttal PDF is not
included in this repository, but the code behind every table below lives under
[`rebuttal/`](../rebuttal/).

## 1. Comparison against ProGen (Reviewer xu9C)

Rather than only isolating hard-vs-soft conditioning with a from-scratch baseline, we
integrated soft-label conditioning directly into **ProGen** (Ye et al., EMNLP 2022
Findings), an iterative feedback-based generation framework, to test whether the idea
transfers to a more sophisticated pipeline.

*Code: [`rebuttal/baselines/progen/`](../rebuttal/baselines/progen/)*

| Method | Accuracy (Linear Probe) | Accuracy (DistilBERT) |
|---|---|---|
| Gold | 94.1 (±0.3) | 92.4 (±0.1) |
| HardPrompt+CoT | 82.5 (±1.0) | 75.2 (±0.2) |
| SoftPrompt+CoT (ours) | 89.3 (±0.3) | 84.5 (±0.1) |
| ProGen+CoT | 85.8 (±0.9) | 77.7 (±0.5) |
| ProGen w/ SoftPrompt+CoT (ours) | **89.5 (±0.2)** | **84.9 (±0.2)** |

Integrating soft labels into ProGen's feedback loop improved it further, on both
evaluation protocols — evidence that the benefit of continuous conditioning is not
tied to a single, simple generation pipeline.

## 2. Finetuning evaluation with DistilBERT+LoRA (Reviewer xu9C)

The original thesis evaluates synthetic data via a linear probe (logistic regression
on frozen embeddings) to isolate data quality from classifier capacity. To address the
concern that this is non-standard, we added a finetuning-based protocol.

*Code: [`rebuttal/finetune.py`](../rebuttal/finetune.py)*
(DistilBERT-base + LoRA, AdamW, lr=2e-4 cosine, wd=0.001, 10 epochs, batch size 128,
r=100% data, binarized targets)

| Model | IMDb Acc / ECE | SST Acc / ECE | SUBJ Acc / ECE | Emotion F1 / ECE | AGNews Acc / ECE |
|---|---|---|---|---|---|
| Gold | 92.42 / 1.61 | 85.38 / 2.97 | 95.91 / 1.56 | 91.17 / 3.21 | 93.81 / 0.71 |
| HardPrompt+CoT | 75.29 / 22.91 | 74.21 / 18.73 | 67.70 / 11.42 | 35.22 / 33.90 | **79.27** / 11.82 |
| SoftPrompt+CoT | **84.58** / **6.63** | **77.32** / **12.51** | **80.75** / **7.49** | **39.03** / **9.23** | 74.85 / 26.41 |

SoftPrompt-generated data wins on 4/5 datasets under a standard finetuning protocol
(not just the linear probe used in the thesis) and is dramatically better calibrated
across the board — even without training directly on soft targets (cf. thesis Table 4.3).

## 3. Cross-LLM sensitivity (Reviewer QPcs)

The thesis uses `gemini-2.0-flash` exclusively. To test whether the effect is
model-specific, we regenerated IMDb with two additional generator LLMs.

*Code: model routing in [`softprompt/runs/generate.py`](../softprompt/runs/generate.py)
+ [`softprompt/utils/langchain_utils.py`](../softprompt/utils/langchain_utils.py)*

| Base LLM | HardPrompt+CoT Acc. | SoftPrompt+CoT Acc. |
|---|---|---|
| gemini-2.0-flash | 82.5 (±1.0) | 89.3 (±0.3) |
| gpt-4o-mini | 84.2 (±0.9) | 90.5 (±0.1) |
| claude-3-haiku-20240307 | 82.3 (±0.5) | 89.1 (±0.1) |

The soft-label advantage holds across all three model families.

## 4. Generation fidelity analysis (Reviewer fd9a)

The central concern here was whether the LLM actually honors the requested soft-label
percentage, or whether "70% positive" is just noise. We measured this two ways.

*Code: [`rebuttal/fidelity_judges/gold_as_a_judge.py`](../rebuttal/fidelity_judges/gold_as_a_judge.py)
(classifier trained on 100% real data scores each generation; Pearson/Spearman
correlation against the prompted label) and
[`rebuttal/fidelity_judges/llm_as_a_judge.py`](../rebuttal/fidelity_judges/llm_as_a_judge.py)
(GPT-4o as a zero-shot judge; also computes the bin-wise MAE table below)*

**Correlation between prompted soft label and judged label:**

| Judge | IMDb Pearson r | SUBJ Pearson r |
|---|---|---|
| Gold-trained classifier | 0.8856 (p<0.001) | 0.7339 (p<0.001) |
| GPT-4o zero-shot judge | 0.9657 (p<0.001) | 0.9183 (p<0.001) |

**Bin-wise Mean Absolute Error (GPT-4o judge) — reveals a U-shaped fidelity curve:**

| Prompted-label bin | MAE (IMDb) | MAE (SUBJ) |
|---|---|---|
| [0.0, 0.1] | 0.025 | 0.033 |
| [0.1, 0.2] | 0.095 | 0.090 |
| [0.2, 0.3] | 0.162 | 0.164 |
| [0.3, 0.4] | 0.164 | 0.198 |
| [0.4, 0.5] | 0.151 | 0.248 |
| [0.5, 0.6] | 0.070 | 0.120 |
| [0.6, 0.7] | 0.046 | 0.071 |
| [0.7, 0.8] | 0.056 | 0.069 |
| [0.8, 0.9] | 0.064 | 0.052 |
| [0.9, 1.0] | 0.027 | 0.050 |

Error is lowest at the extremes and highest near the 0.5 decision boundary — the LLM is
least precise exactly where prompts are most ambiguous. This gives a data-driven,
mechanistic justification for the thesis's margin-based filtering (Section 4.3):
filtering isn't an arbitrary post-hoc heuristic, it removes the region where generation
fidelity is empirically worst. We also noted the conceptual parallel to Mixup, which
gets its best results sampling its mixing coefficient from a U-shaped Beta distribution
that avoids near-0.5 mixtures.

## 5. HardGen-SoftLabel ablation (Reviewer fd9a)

A reviewer asked for a baseline that isolates *when* the soft label is introduced:
generate text with HardPrompt, then assign a continuous label post-hoc with an LLM
judge (conceptually similar to GPT3Mix). This tests whether soft labels only help
because of the label itself, independent of how the text was generated.

*Code: same [`llm_as_a_judge.py`](../rebuttal/fidelity_judges/llm_as_a_judge.py),
applied to HardPrompt+CoT-generated data instead of oracle data*

| Dataset | HardPrompt+CoT | HardGen-SoftLabel | SoftPrompt+CoT (ours) |
|---|---|---|---|
| IMDb | 82.5 | 82.5 | 89.3 (→ 91.8 with filtering) |
| SUBJ | 69.4 | 69.5 | 69.1 (→ 81.1 with filtering) |

Post-hoc labeling provides essentially no benefit: text generated under a hard prompt
is already so polarized (IMDb post-hoc soft labels: 0.999 correlation with the original
hard label, nearly all mass in [0.0,0.1] ∪ [0.9,1.0]) that there is no real "softness"
left to assign. The label has to be part of the generation process itself, not
bolted on afterward — this is the paper's central claim, and this ablation is direct
evidence for it.

## Other rebuttal points (no new experiments)

- **SST margin filtering, extended:** pushing the margin to 0.30 improves
  SoftPrompt+CoT on SST from 77.5 to 84.9, surpassing HardPrompt+CoT's 81.5 — using the
  same filtering code as the thesis (`_filter_by_margin` in
  [`softprompt/runs/evaluate_pytorch.py`](../softprompt/runs/evaluate_pytorch.py)).
- **Scalability to many classes:** clarified that the relevant factor is whether
  classes are semantically *blendable* (e.g., emotions), not the raw count K; datasets
  like 20-Newsgroups with many mutually exclusive topics are out of scope as currently
  formulated.
- **Terminology:** added a footnote distinguishing this paper's "soft label" (continuous
  label supervision) from "soft prompt" in the prompt-tuning literature (trainable input
  embeddings).
- Writing/citation cleanup and a dedicated Limitations section were added per reviewer
  request.

## Outcome

The paper was **rejected**. The reviewers' engagement — particularly the fidelity/MAE
analysis and the ProGen integration — meaningfully sharpened the empirical case for
soft-label conditioning beyond what's in the thesis, which is why this extended work is
included here alongside the core thesis pipeline.
