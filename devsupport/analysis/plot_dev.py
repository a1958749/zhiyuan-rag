"""Render observed DEV metrics; never substitute them for holdout or answer accuracy."""

import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
base = ROOT / "devsupport/reports"
labels = ["B0", "H0", "H1", "R1", "R2", "C"]
coverage = []
for name in labels:
    folder = "contextual-v1" if name == "C" else "rerank-v1" if name.startswith("R") else "retrieval-v1"
    value = json.loads((base / folder / "dev" / f"{name}.summary.json").read_text(encoding="utf-8"))
    coverage.append(value.get("metrics", value.get("metrics_with_fallback"))["coverage@10"] * 100)
packed = {}
for folder in ["packing-budget-comparison", "packing-rerank-dev", "packing-contextual-dev"]:
    summary = json.loads((base / folder / "summary.json").read_text(encoding="utf-8"))
    for name, item in summary["methods"].items():
        packed[name] = item["policies"]["cl100k_base_proxy_v1"]["packed_nugget_coverage"] * 100
colors = ["#b0bbc9", "#748ca9", "#748ca9", "#8495aa", "#8495aa", "#167d9a"]
fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.8))
for ax, values, title in [
    (axes[0], coverage, "Retrieved evidence: Coverage@10"),
    (axes[1], [packed[n] for n in labels], "Evidence sent to the model: coverage"),
]:
    ax.bar(labels, values, color=colors, width=0.62)
    ax.set_title(title, fontsize=12, pad=15, loc="left")
    ax.set_ylim(0, 65)
    ax.set_ylabel("Nugget coverage (%)")
    ax.grid(axis="y", alpha=0.18)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    for x, y in enumerate(values):
        ax.text(x, y + 1.2, f"{y:.1f}", ha="center", fontsize=10)
fig.suptitle(
    "DevSupport Copilot | Development-set evidence audit", x=0.065, ha="left", fontsize=17, fontweight="bold", y=0.98
)
fig.text(
    0.065, 0.10, "DEV n=53; full 49,505-block corpus; shared 4,000 cl100k proxy-token context budget.", fontsize=10
)
fig.text(
    0.065, 0.062, "C minus B0 Coverage@10: +9.52 pp; paired bootstrap 95% interval: +0.85 to +19.12 pp.", fontsize=10
)
fig.text(
    0.065,
    0.022,
    "Development results only. These are evidence metrics, not answer accuracy or holdout results.",
    fontsize=10,
    color="#7a451e",
)
fig.subplots_adjust(top=0.83, bottom=0.25, wspace=0.28, left=0.065, right=0.98)
out = base / "figures"
out.mkdir(exist_ok=True)
fig.savefig(out / "dev-evidence-audit.png", dpi=160)
fig.savefig(out / "dev-evidence-audit.svg")
print(out / "dev-evidence-audit.png")
