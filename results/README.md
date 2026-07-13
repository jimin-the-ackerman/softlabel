# Results

`tables/` contains the full numeric results reported in the thesis, transcribed
directly from the thesis PDF (Table 4.2, Appendix Tables A.1/A.2, Table 4.3, Tables
4.4–4.7). `figures/figure_4_1_margin_filtering.png` is a reproduction of Figure 4.1,
generated from `appendix_a1/a2` via `figures/make_figure_4_1.py` — the original figure
was produced by `notebooks/additional_analysis/0_margin_filtering_plot_figures.ipynb`.

**Not included:** the raw per-bootstrap-trial outputs (`results/{dataset}/{model}/{run}/`
directories with `data.jsonl`, `embeddings/`, `config.yaml` etc. for every generation
run behind these tables) are not tracked in git — they were multiple GB across 5
datasets × 6 data-size ratios × 2 (Hard/Soft) × 2 (CoT) × 50 bootstrap trials. See
[`examples/`](../examples/) for small, real samples of the generation format instead.
