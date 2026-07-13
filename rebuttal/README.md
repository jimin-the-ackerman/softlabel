# NeurIPS 2025 Rebuttal Extensions

Code added during the NeurIPS 2025 rebuttal period, on top of the thesis-era
[`softprompt/`](../softprompt/) package. See [`docs/rebuttal_summary.md`](../docs/rebuttal_summary.md)
for what each piece was used to answer.

- `finetune.py` — DistilBERT+LoRA finetuning evaluation protocol (Table 2 in the rebuttal).
- `fidelity_judges/gold_as_a_judge.py` — scores generated text with a classifier trained
  on 100% real data; Pearson/Spearman correlation against the prompted soft label.
- `fidelity_judges/llm_as_a_judge.py` — scores generated text with GPT-4o as a zero-shot
  judge; also computes the bin-wise MAE table (the U-shaped fidelity curve). Applying
  this to HardPrompt-generated data (instead of SoftPrompt-generated data) is exactly
  the `HardGen-SoftLabel` ablation in the rebuttal.
- `baselines/progen/` — a reimplementation of ProGen (Ye et al., EMNLP 2022 Findings),
  extended to accept soft-label-conditioned generation instead of only hard labels.

**`loaders.py`** in this folder is a separate, later version of the dataset loader used
by the code above (class-based `IMDb`/`SUBJ`/`SST`/`Emotion`/`AGNews`/`Yahoo` loaders,
vs. the thesis-era `load_oracle_data`/`load_synthetic_data` functions still used by
`softprompt/runs/evaluate_pytorch.py`). The two were never unified in the original
working repository, so they're kept separate here rather than force-merged.
