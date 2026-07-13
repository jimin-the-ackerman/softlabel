import pandas as pd
import matplotlib.pyplot as plt

no_cot = pd.read_csv("results/tables/appendix_a1_margin_filtering_no_cot.csv")
with_cot = pd.read_csv("results/tables/appendix_a2_margin_filtering_with_cot.csv")

margins = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
margin_cols = [f"m{m:.2f}" for m in margins]

datasets = ["imdb", "sst", "subj"]
titles = {"imdb": "IMDb", "sst": "SST", "subj": "SUBJ"}
sizes = ["1%", "5%", "10%", "25%", "50%", "100%"]
colors = plt.cm.viridis_r([i / (len(sizes) - 1) for i in range(len(sizes))])

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

for ax, ds in zip(axes, datasets):
    df_nc = no_cot[no_cot["dataset"] == ds]
    df_c = with_cot[with_cot["dataset"] == ds]
    for size, color in zip(sizes, colors):
        row_nc = df_nc[df_nc["size"] == size]
        row_c = df_c[df_c["size"] == size]
        if len(row_nc):
            ax.plot(margins, row_nc[margin_cols].values.flatten(),
                    color=color, linestyle="-", marker="o", markersize=3, linewidth=1.3,
                    label=f"{size}" if ds == "imdb" else None)
        if len(row_c):
            ax.plot(margins, row_c[margin_cols].values.flatten(),
                    color=color, linestyle="--", marker="o", markersize=3, linewidth=1.3)
    ax.set_title(titles[ds])
    ax.set_xlabel("Margin (m)")
    if ds == "imdb":
        ax.set_ylabel("Accuracy (%)")
    ax.grid(alpha=0.25)

# Legend: sizes (color) + CoT (linestyle)
from matplotlib.lines import Line2D
size_handles = [Line2D([0], [0], color=c, lw=2, label=s) for s, c in zip(sizes, colors)]
style_handles = [
    Line2D([0], [0], color="black", lw=1.5, linestyle="-", label="No CoT"),
    Line2D([0], [0], color="black", lw=1.5, linestyle="--", label="CoT"),
]
fig.legend(handles=size_handles + style_handles, loc="lower center", ncol=8,
           bbox_to_anchor=(0.5, -0.08), frameon=False, fontsize=9)

fig.suptitle("Figure 4.1 (reproduced): Impact of margin-based filtering on\ndownstream classification accuracy (binary tasks)", fontsize=11)
fig.tight_layout(rect=[0, 0.05, 1, 0.92])
fig.savefig("results/figures/figure_4_1_margin_filtering.png", dpi=200, bbox_inches="tight")
fig.savefig("results/figures/figure_4_1_margin_filtering.pdf", bbox_inches="tight")
print("saved.")
