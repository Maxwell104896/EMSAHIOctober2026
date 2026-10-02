import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("Fig4B_state_composition_long.csv")
plot_order = ["non-IBD", "CD", "UC"]
state_order = ["low-like", "intermediate", "high-like"]

wide = (
    df.pivot(index="diagnosis_display", columns="state_display", values="percent")
      .loc[plot_order, state_order]
)

fig, ax = plt.subplots(figsize=(6.4, 4.6))
colors = ["#7EA6E0", "#F0C15A", "#7BC17C"]
bottom = np.zeros(len(plot_order))
for state, color in zip(state_order, colors):
    vals = wide[state].to_numpy()
    ax.bar(plot_order, vals, bottom=bottom, label=state, color=color,
           edgecolor="white", linewidth=0.8)
    for i, v in enumerate(vals):
        ax.text(i, bottom[i] + v/2, f"{v:.0f}%", ha="center", va="center", fontsize=10)
    bottom += vals

ax.set_ylim(0, 100)
ax.set_ylabel("Percentage of broad-score visits")
ax.set_title("Broad molecular-state composition")
ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.12))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
fig.savefig("Fig4B_reference.png", dpi=600, bbox_inches="tight")
fig.savefig("Fig4B_reference.svg", bbox_inches="tight")
plt.show()
