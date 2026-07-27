"""UAV-Embodied-IQA paper figures — 4-VLM aggregated data.

Design references:
  Embodied-IQA  Fig 5  -> fig_distribution  (5-level color coding per distortion)
  Embodied-IQA  Fig 7  -> fig_intensity_curves (6-panel intensity × mean, ±SEM)
  Embodied-IQA  Fig 4  -> fig_intensity_fingerprint (Spearman rho heatmap + avg label)
  AirCopBench   Fig 5  -> fig_task_distortion (4-group separator, mean per cell)

Run:  uv run python figures/gen_all_figures.py
"""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import json
from pathlib import Path
from collections import defaultdict, Counter
from scipy.stats import spearmanr

from paper_plot_style import FIG_DIR, DATA_DIR, CB_PALETTE, COLORS, save_fig, FONT_SIZE

# ── Load ──────────────────────────────────────────────────────────
data_dir = DATA_DIR / "annotated" / "test"
all_data = []
for f in sorted(data_dir.glob("*.json")):
    with open(f, encoding="utf-8") as fp:
        all_data.extend(json.load(fp))
print("Loaded %d test entries" % len(all_data))

# ── Constants ─────────────────────────────────────────────────────
UAV_DISTS = [
    "propeller_vibration_blur", "atmospheric_scattering_haze",
    "six_dof_viewpoint_blur",   "communication_packet_loss",
    "low_res_super_resolution", "propeller_shadow",
]
UAV_FULL = {
    "propeller_vibration_blur":    "Prop. Vibration",
    "atmospheric_scattering_haze": "Atm. Scattering",
    "six_dof_viewpoint_blur":      "6DoF Blur",
    "communication_packet_loss":   "Pkt-Loss",
    "low_res_super_resolution":    "LR+SR",
    "propeller_shadow":            "Prop. Shadow",
}
# Embodied-IQA GroupA–E level palette
LEVEL_COLORS = {
    2:  "#92ca2c",
    4:  "#3c7daa",
    6:  "#e64f04",
    8:  "#f2c400",
    10: "#a757a7",
}
LEVELS     = [2, 4, 6, 8, 10]
VLM_ORDER  = ["Qwen2.5-VL", "Qwen2-VL", "InternVL2", "Mini-InternVL"]
VLM_SHORT  = ["Q2.5", "Q2", "IV2", "Mini"]

# AirCopBench task-dimension grouping
TASK_GROUPS = {
    "Scene\nUnderstanding":   ["Scene Description", "Scene Comparison", "Observing Posture"],
    "Object\nUnderstanding":  ["Object Recognition", "Object Grounding",
                               "Object Matching",    "Object Counting"],
    "Perception\nAssessment": ["Quality Assessment", "Usability Assessment", "Causal Assessment"],
    "Collaboration":          ["Who to Collaborate", "What to Collaborate",
                               "Why to Collaborate", "When to Collaborate"],
}
QTYPE_ORDER = []
for _members in TASK_GROUPS.values():
    QTYPE_ORDER.extend(_members)

# ── Pre-index ─────────────────────────────────────────────────────
dist_data  = defaultdict(list)
level_data = defaultdict(lambda: defaultdict(list))
for e in all_data:
    d  = e["distortion_info"]["type"]
    lv = e["distortion_info"]["level"]
    dist_data[d].append(e)
    level_data[d][lv].append(e["cognitive_score"])

all_dists_sorted = sorted(
    dist_data.keys(),
    key=lambda d: np.mean([e["cognitive_score"] for e in dist_data[d]])
)

# ─────────────────────────────────────────────────────────────────
# Fig 1  fig_distribution
# Embodied-IQA Fig 5 style:
#   x = 36 distortion types sorted by mean cognitive score
#   y = cognitive score
#   color = distortion level (L2/4/6/8/10), 5 scatter dots per distortion
#   UAV distortions: orange x-tick labels; dashed vertical separators
# ─────────────────────────────────────────────────────────────────
def fig_distribution():
    fig, ax = plt.subplots(figsize=(12, 3.6))

    for xi, d in enumerate(all_dists_sorted):
        for lv in LEVELS:
            scores = level_data[d][lv]
            if not scores:
                continue
            mean_v = np.mean(scores)
            ax.scatter(xi, mean_v, color=LEVEL_COLORS[lv],
                       s=18, alpha=0.85, linewidths=0, zorder=3)
        # overall mean marker
        all_scores = [e["cognitive_score"] for e in dist_data[d]]
        ax.scatter(xi, np.mean(all_scores), color="black",
                   s=10, marker="D", zorder=5)

    # x-axis labels — UAV orange, generic gray
    short_labels = [UAV_FULL.get(d, d.replace("_", " ")) for d in all_dists_sorted]
    ax.set_xticks(range(len(all_dists_sorted)))
    ax.set_xticklabels(short_labels, rotation=55, ha="right", fontsize=5.5)
    for i, (tick, d) in enumerate(zip(ax.get_xticklabels(), all_dists_sorted)):
        tick.set_color(COLORS["uav"] if d in UAV_DISTS else "#444444")

    # UAV group background shading
    uav_idxs = [i for i, d in enumerate(all_dists_sorted) if d in UAV_DISTS]
    for xi in uav_idxs:
        ax.axvspan(xi - 0.45, xi + 0.45, color=COLORS["uav"], alpha=0.08, zorder=0)

    ax.set_ylabel("Cognitive Score", fontsize=FONT_SIZE)
    ax.set_xlim(-0.6, len(all_dists_sorted) - 0.4)
    ax.axhline(np.mean([e["cognitive_score"] for e in all_data]),
               color="#888888", lw=0.8, ls="--", label="Grand mean")

    # Level legend (Embodied-IQA style)
    handles = [mpatches.Patch(color=LEVEL_COLORS[lv], label="Level %d" % lv) for lv in LEVELS]
    handles.append(plt.Line2D([0], [0], marker="D", color="black", ls="none",
                               markersize=4, label="Overall mean"))
    ax.legend(handles=handles, frameon=False, fontsize=7.5,
              ncol=3, loc="upper left", handlelength=1.2)

    plt.tight_layout(pad=0.5)
    save_fig(fig, "fig_distribution")


# ─────────────────────────────────────────────────────────────────
# Fig 2  fig_intensity_curves
# Embodied-IQA Fig 6 style:
#   2×3 grid, one panel per UAV distortion
#   x = intensity level, y = mean cognitive score
#   single UAV-orange line + ±SEM shading (not error bars)
#   shared y-axis range per row for easy comparison
# ─────────────────────────────────────────────────────────────────
def fig_intensity_curves():
    fig, axes = plt.subplots(2, 3, figsize=(9, 4.2), sharey="row")
    axes = axes.flatten()

    for idx, d in enumerate(UAV_DISTS):
        ax = axes[idx]
        means = np.array([np.mean(level_data[d][lv]) for lv in LEVELS])
        sems  = np.array([np.std(level_data[d][lv]) / np.sqrt(len(level_data[d][lv]))
                          for lv in LEVELS])
        xs = np.array(LEVELS)

        ax.fill_between(xs, means - sems, means + sems,
                        color=COLORS["uav"], alpha=0.20)
        ax.plot(xs, means, "-o", color=COLORS["uav"],
                lw=1.8, markersize=5, markerfacecolor="white",
                markeredgewidth=1.5, zorder=4)

        # Level-colored dots on top
        for lv, m in zip(LEVELS, means):
            ax.plot(lv, m, "o", color=LEVEL_COLORS[lv], markersize=5, zorder=5)

        ax.set_title(UAV_FULL[d], fontsize=FONT_SIZE - 1, pad=3, color=COLORS["uav"])
        ax.set_xticks(LEVELS)
        ax.set_xticklabels(["L%d" % lv for lv in LEVELS], fontsize=7)
        ax.tick_params(axis="y", labelsize=7)
        if idx in (0, 3):
            ax.set_ylabel("Cognitive Score", fontsize=8)
        if idx >= 3:
            ax.set_xlabel("Distortion Level", fontsize=8)

    plt.tight_layout(pad=0.8)
    save_fig(fig, "fig_intensity_curves")


# ─────────────────────────────────────────────────────────────────
# Fig 3  fig_intensity_fingerprint
# Embodied-IQA Fig 4 style:
#   rows = 36 distortions (sorted by mean), cols = 4 VLMs
#   cell = Spearman ρ(score, intensity level)
#   diverging RdYlGn; average ρ per VLM shown below x-axis
#   UAV rows: orange left-margin strip
# ─────────────────────────────────────────────────────────────────
def fig_intensity_fingerprint():
    n_d = len(all_dists_sorted)
    rho_mat = np.zeros((n_d, len(VLM_ORDER)))
    for di, d in enumerate(all_dists_sorted):
        entries = dist_data[d]
        lvs = [e["distortion_info"]["level"] for e in entries]
        for vi, v in enumerate(VLM_ORDER):
            sc = [e["vlm_scores"].get(v, float("nan")) for e in entries]
            r, _ = spearmanr(lvs, sc)
            rho_mat[di, vi] = r

    fig, ax = plt.subplots(figsize=(4.2, 9.5))
    im = ax.imshow(rho_mat, vmin=-0.6, vmax=0.6,
                   cmap="RdYlGn", aspect="auto", interpolation="nearest")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.08,
                        orientation="horizontal")
    cbar.set_label(r"Spearman $\rho$  (score vs. intensity level)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    row_labels = [UAV_FULL.get(d, d.replace("_", " ")) for d in all_dists_sorted]
    ax.set_yticks(range(n_d))
    ax.set_yticklabels(row_labels, fontsize=6)
    ax.set_xticks(range(len(VLM_ORDER)))
    ax.set_xticklabels(VLM_SHORT, fontsize=8)

    # UAV left-margin strip (outside plot: use broken_barh in axes coords)
    for di, d in enumerate(all_dists_sorted):
        if d in UAV_DISTS:
            ax.add_patch(plt.Rectangle((-0.5 - 0.35, di - 0.5), 0.3, 1.0,
                                       color=COLORS["uav"], clip_on=False, zorder=6))

    # Annotate cells
    for di in range(n_d):
        for vi in range(len(VLM_ORDER)):
            v = rho_mat[di, vi]
            fc = "white" if abs(v) > 0.4 else "black"
            ax.text(vi, di, "%.2f" % v, ha="center", va="center",
                    fontsize=5, color=fc)

    # Per-VLM average rho below x-axis (Embodied-IQA style)
    for vi, v in enumerate(VLM_ORDER):
        avg = np.mean(rho_mat[:, vi])
        ax.text(vi, n_d + 0.5, "avg=%.2f" % avg,
                ha="center", va="bottom", fontsize=6.5,
                color="#333333", transform=ax.transData)

    ax.set_xlim(-0.5 - 0.5, len(VLM_ORDER) - 0.5)
    plt.tight_layout(pad=0.5)
    save_fig(fig, "fig_intensity_fingerprint")


# ─────────────────────────────────────────────────────────────────
# Fig 4  fig_task_distortion
# AirCopBench Fig 5 style:
#   rows = 14 question types, grouped into 4 task dimensions
#   cols = 6 UAV distortions
#   cell = raw mean cognitive score (not residual — reviewer noted residual is
#          confusing to readers; raw mean with row/col annotation is cleaner)
#   group separators + group labels on left (as in AirCopBench correlation fig)
#   darker = more severe cognitive degradation (reversed colormap)
# ─────────────────────────────────────────────────────────────────
def fig_task_distortion():
    # Build ordered row list matching QTYPE_ORDER (drop missing)
    present_qtypes = set(e["question_type"] for e in all_data)
    ordered_qt = [q for q in QTYPE_ORDER if q in present_qtypes]
    n_q = len(ordered_qt)
    n_u = len(UAV_DISTS)

    cell_scores = defaultdict(list)
    for e in all_data:
        if e["distortion_info"]["type"] in UAV_DISTS:
            cell_scores[(e["question_type"], e["distortion_info"]["type"])].append(
                e["cognitive_score"])

    mat = np.full((n_q, n_u), np.nan)
    for qi, qt in enumerate(ordered_qt):
        for di, d in enumerate(UAV_DISTS):
            vals = cell_scores.get((qt, d), [])
            if vals:
                mat[qi, di] = np.mean(vals)

    fig, ax = plt.subplots(figsize=(5.8, 6.2))
    # reversed: lower score = darker red
    im = ax.imshow(mat, vmin=0.5, vmax=1.1, cmap="RdYlGn", aspect="auto")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Mean Cognitive Score", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    ax.set_xticks(range(n_u))
    ax.set_xticklabels([UAV_FULL[d] for d in UAV_DISTS],
                       rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(n_q))
    ax.set_yticklabels(ordered_qt, fontsize=7)

    # Annotate cells
    for qi in range(n_q):
        for di in range(n_u):
            v = mat[qi, di]
            if not np.isnan(v):
                ax.text(di, qi, "%.2f" % v, ha="center", va="center",
                        fontsize=6,
                        color="white" if v < 0.65 or v > 1.0 else "black")

    # Task-group separators and left labels (AirCopBench style)
    row_cursor = 0
    grp_colors = ["#E69F00", "#56B4E9", "#009E73", "#CC79A7"]
    for gi, (grp_name, members) in enumerate(TASK_GROUPS.items()):
        n_m = len([m for m in members if m in present_qtypes])
        if n_m == 0:
            continue
        # horizontal separator
        sep_y = row_cursor - 0.5
        if gi > 0:
            ax.axhline(sep_y, color="#333333", lw=1.0, ls="--", alpha=0.6)
        # group label on left side
        mid_y = row_cursor + n_m / 2.0 - 0.5
        ax.text(-0.6, mid_y, grp_name, ha="right", va="center",
                fontsize=7, color=grp_colors[gi],
                fontweight="bold", transform=ax.transData)
        # colored left bar
        ax.add_patch(plt.Rectangle((-0.5 - 0.25, row_cursor - 0.5),
                                   0.18, n_m, color=grp_colors[gi],
                                   clip_on=False, zorder=6))
        row_cursor += n_m

    ax.set_xlim(-0.5 - 0.5, n_u - 0.5)
    plt.tight_layout(pad=0.5)
    save_fig(fig, "fig_task_distortion")


# ─────────────────────────────────────────────────────────────────
# Fig 5  fig_vlm_corr
# Embodied-IQA Fig 4 style:
#   4x4 SRCC matrix of VLM annotators
#   diagonal boxes outlined; per-column avg below x-axis
# ─────────────────────────────────────────────────────────────────
def fig_vlm_corr():
    n = len(VLM_ORDER)
    scores = {v: [] for v in VLM_ORDER}
    for e in all_data:
        for v in VLM_ORDER:
            scores[v].append(e["vlm_scores"].get(v, float("nan")))

    srcc_mat = np.zeros((n, n))
    for i, v1 in enumerate(VLM_ORDER):
        for j, v2 in enumerate(VLM_ORDER):
            r, _ = spearmanr(scores[v1], scores[v2])
            srcc_mat[i, j] = r

    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    im = ax.imshow(srcc_mat, vmin=0.0, vmax=1.0, cmap="Blues", aspect="auto")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("SRCC", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(VLM_ORDER, rotation=40, ha="right", fontsize=7)
    ax.set_yticklabels(VLM_ORDER, fontsize=7)

    for i in range(n):
        for j in range(n):
            val = srcc_mat[i, j]
            color = "white" if val > 0.6 else "black"
            weight = "bold" if i == j else "normal"
            ax.text(j, i, "%.2f" % val, ha="center", va="center",
                    fontsize=7, color=color, fontweight=weight)

    # diagonal outline
    for k in range(n):
        ax.add_patch(plt.Rectangle((k - 0.5, k - 0.5), 1, 1,
                                   fill=False, edgecolor="#333333", lw=1.2))

    off_diag = srcc_mat[~np.eye(n, dtype=bool)]
    ax.set_xlabel("Average inter-model SRCC = %.3f" % off_diag.mean(), fontsize=8)

    plt.tight_layout(pad=0.5)
    save_fig(fig, "fig_vlm_corr")


# ─────────────────────────────────────────────────────────────────
# Fig 6  fig_dataset_stats
# AirCopBench Fig 3 style:
#   3-panel: (a) score by distortion category (bar, mean±std)
#            (b) sample count by question type (horizontal bar)
#            (c) sample count by dataset source (vertical bar)
# ─────────────────────────────────────────────────────────────────
def fig_dataset_stats():
    CAT_ORDER  = ["UAV", "Blur", "Noise", "Luminance", "Chrominance",
                  "Compression", "Spatial", "Other"]
    CAT_COLORS = {
        "UAV": COLORS["uav"], "Blur": CB_PALETTE[1], "Noise": CB_PALETTE[4],
        "Luminance": CB_PALETTE[5], "Chrominance": CB_PALETTE[2],
        "Compression": CB_PALETTE[3], "Spatial": CB_PALETTE[6], "Other": "#AAAAAA",
    }
    RAW_TO_CAT = {
        "uav": "UAV", "blur": "Blur", "noise": "Noise",
        "brightness": "Luminance", "chromatic": "Chrominance",
        "compression": "Compression", "spatial": "Spatial",
        "transmission": "Other", "other": "Other",
    }

    cat_scores = defaultdict(list)
    for e in all_data:
        cat = RAW_TO_CAT.get(e["distortion_info"]["category"], "Other")
        cat_scores[cat].append(e["cognitive_score"])

    qtype_counts = Counter(e["question_type"] for e in all_data)
    qt_sorted = sorted(qtype_counts.items(), key=lambda x: -x[1])
    qt_labels = [k for k, _ in qt_sorted]
    qt_vals   = [v for _, v in qt_sorted]

    ds_counts = Counter(e["dataset"] for e in all_data)
    ds_order  = sorted(ds_counts.keys())

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))

    # (a) mean ± std bar — cleaner than box for category-level overview
    ax = axes[0]
    present = [c for c in CAT_ORDER if c in cat_scores]
    means = [np.mean(cat_scores[c]) for c in present]
    stds  = [np.std(cat_scores[c])  for c in present]
    colors_a = [CAT_COLORS[c] for c in present]
    bars = ax.bar(range(len(present)), means, color=colors_a, alpha=0.82, width=0.6)
    ax.errorbar(range(len(present)), means, yerr=stds,
                fmt="none", color="#333333", capsize=3, lw=0.9)
    ax.set_xticks(range(len(present)))
    ax.set_xticklabels(present, rotation=35, ha="right", fontsize=7)
    ax.set_ylabel("Cognitive Score", fontsize=8)
    ax.set_xlabel("(a) Score by Distortion Category", fontsize=8)
    grand = np.mean([e["cognitive_score"] for e in all_data])
    ax.axhline(grand, color="gray", ls="--", lw=0.8, alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # (b) horizontal bar — question type
    ax = axes[1]
    ypos = range(len(qt_labels))
    ax.barh(ypos, qt_vals, color=CB_PALETTE[1], alpha=0.82)
    ax.set_yticks(ypos)
    ax.set_yticklabels(qt_labels, fontsize=7)
    ax.set_xlabel("Number of Samples", fontsize=8)
    ax.set_ylabel("(b) Question Type Distribution", fontsize=8)
    ax.invert_yaxis()
    for i, val in enumerate(qt_vals):
        ax.text(val + 30, i, str(val), va="center", fontsize=6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # (c) vertical bar — dataset source
    ax = axes[2]
    ds_vals   = [ds_counts[d] for d in ds_order]
    colors_ds = [CB_PALETTE[i % len(CB_PALETTE)] for i in range(len(ds_order))]
    bars = ax.bar(ds_order, ds_vals, color=colors_ds, alpha=0.85, width=0.5)
    ax.set_xlabel("(c) Dataset Source", fontsize=8)
    ax.set_ylabel("Number of Samples", fontsize=8)
    for bar, val in zip(bars, ds_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 40,
                str(val), ha="center", va="bottom", fontsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout(pad=1.0)
    save_fig(fig, "fig_dataset_stats")
if __name__ == "__main__":
    fig_distribution()
    fig_intensity_curves()
    fig_intensity_fingerprint()
    fig_task_distortion()
    fig_vlm_corr()
    fig_dataset_stats()
    print("\nAll figures saved to figures/")
