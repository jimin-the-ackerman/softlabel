# Examples

Real SoftPrompt+CoT generation runs, one per dataset (`gemini-2.0-flash`, truncated to
200 records each to keep the repo small; `config.yaml` in each folder shows the full
run configuration used to produce the untruncated version).

Each line of `data.jsonl` is a `{"label": ..., "text": ..., "reasoning": ...}` record:
`label` is the sampled soft label (a scalar in `[0,1]` for the binary tasks — imdb,
sst, subj — or a probability vector for the multi-class tasks — emotion, agnews),
`text` is the generated example, and `reasoning` is the model's CoT trace explaining
how it translated the label into the text.

The IMDb sample here is the exact source of the "Moderately Positive" example
reproduced in the thesis's Appendix D (Table D.1, soft label 0.77).
