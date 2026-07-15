"""Generate all dataset analysis figures for UAV-Embodied-IQA paper.
Replicates Embodied-IQA Section 4 (Database Analysis) methodology:
  1. Distribution of cognitive scores across 36 distortion types
  2. UAV vs Generic distortion comparison bar chart
  3. 3-dimensional score breakdown (BLEU/ROUGE-L/CIDEr)
  4. JND-based sensitivity classification heatmap
  5. Per-intensity distortion degradation curves
  6. Source comparison (Sim vs Real)

All output saved to figures/ directory.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path
from collections import defaultdict

# ── Configuration ────────────────────────────────────────────────
FIG_DIR = Path(__file__).resolve().parent
DATA_DIR = FIG_DIR.parent / "data" / "annotated" / "vlm" / "Qwen2-VL"
FONT_SIZE = 9
DPI = 300

matplotlib.rcParams.update({
    'font.size': FONT_SIZE,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'axes.labelsize': FONT_SIZE,
    'xtick.labelsize': FONT_SIZE - 1,
    'ytick.labelsize': FONT_SIZE - 1,
    'legend.fontsize': FONT_SIZE - 1,
    'figure.dpi': DPI,
    'savefig.dpi': DPI,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.grid': False,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'text.usetex': False,
})

# Color palette (colorblind-safe)
C_UAV     = '#E69F00'  # orange
C_GENERIC = '#56B4E9'  # sky blue
C_REAL    = '#009E73'  # green
C_SIM     = '#CC79A7'  # purple
C_DIAMOND = '#000000'
CB6 = ['#E69F00', '#56B4E9', '#009E73', '#CC79A7', '#D55E00', '#0072B2']

UAV_DISTS = [
    'propeller_vibration_blur', 'atmospheric_scattering_haze',
    'six_dof_viewpoint_blur', 'communication_packet_loss',
    'low_res_super_resolution', 'propeller_shadow',
]

UAV_DISPLAY = {
    'propeller_vibration_blur':    'Propeller Vibration',
    'atmospheric_scattering_haze': 'Atmospheric Scattering',
    'six_dof_viewpoint_blur':      '6DoF Viewpoint Blur',
    'communication_packet_loss':   'Packet-Loss Blocks',
    'low_res_super_resolution':    'Low-Res + SR',
    'propeller_shadow':            'Propeller Shadow',
}

DIST_CATEGORY = {
    'propeller_vibration_blur':    'UAV',
    'atmospheric_scattering_haze': 'UAV',
    'six_dof_viewpoint_blur':     'UAV',
    'communication_packet_loss':  'UAV',
    'low_res_super_resolution':   'UAV',
    'propeller_shadow':           'UAV',
    'gaussian_blur':     'Blur',  'lens_blur':         'Blur',
    'motion_blur':       'Blur',
    'brighten_max':      'Luminance', 'brighten_avg':  'Luminance',
    'darken_max':        'Luminance', 'darken_min':    'Luminance',
    'darken_avg':        'Luminance',
    'color_diffusion':   'Chrominance', 'color_shift': 'Chrominance',
    'color_quantize':    'Chrominance',
    'white_noise':       'Noise',       'color_noise': 'Noise',
    'impulse_noise':     'Noise',       'multiplicative_noise': 'Noise',
    'gaussian_denoise':  'Noise',       'cnn_denoise': 'Noise',
    'jpeg_compression':  'Compression', 'jp2k_compression': 'Compression',
    'webp_compression':  'Compression',
    'spatial_warp':      'Spatial',     'spatial_scale': 'Spatial',
    'clock_jittering':   'Spatial',     'resolution_limit': 'Spatial',
    'grayscale':         'Other',       'sharpness':     'Other',
    'contrast':          'Other',
    'block_lost':        'Other',       'block_interpolation': 'Other',
    'block_exchange':    'Other',
}

CATEGORY_COLORS = {
    'UAV': '#E69F00', 'Blur': '#56B4E9', 'Luminance': '#F0E442',
    'Chrominance': '#009E73', 'Noise': '#CC79A7',
    'Compression': '#D55E00', 'Spatial': '#0072B2', 'Other': '#888888',
}


def save_fig(fig, name):
    path = FIG_DIR / f"{name}.pdf"
    fig.savefig(str(path))
    print(f"  -> {path}")
    plt.close(fig)


# ── Data Loading ─────────────────────────────────────────────────
def load_data():
    data = []
    for split in ['train', 'test']:
        sd = DATA_DIR / split
        if sd.exists():
            for fpath in sorted(sd.glob('*.json')):
                with open(fpath) as f:
                    data.extend(json.load(f))
    return data


def parse_sid(sid):
    parts = sid.split('__')
    dist_str = parts[4] if len(parts) > 4 else ''
    dist_name, intensity = dist_str, None
    if '_L' in dist_str:
        idx = dist_str.rfind('_L')
        dist_name = dist_str[:idx]
        try:
            intensity = int(dist_str[idx+2:]) / 10.0
        except ValueError:
            intensity = None
    return {
        'source': parts[0] if len(parts) > 0 else '',
        'split':  parts[1] if len(parts) > 1 else '',
        'task':   parts[3] if len(parts) > 3 else '',
        'distortion': dist_name,
        'intensity': intensity,
    }


def cognitive(item):
    return item['bleu'] + item['rouge_l'] + 0.1 * item['cider']


# ── Figure 1: Distribution Box Plot ──────────────────────────────
def fig1_distribution(data):
    print("Figure 1: Cognitive score distribution across 36 distortion types")
    dist_scores = defaultdict(list)
    for item in data:
        info = parse_sid(item['sample_id'])
        dist_scores[info['distortion']].append(cognitive(item))

    all_dists = sorted(dist_scores.keys(),
                       key=lambda d: (0 if d in UAV_DISTS else 1, d))
    uav_count = sum(1 for d in all_dists if d in UAV_DISTS)

    labels = [UAV_DISPLAY.get(d, d.replace('_', ' ').title()) for d in all_dists]
    box_data = [dist_scores[d] for d in all_dists]
    colors = [C_UAV if d in UAV_DISTS else C_GENERIC for d in all_dists]
    means = [np.mean(v) for v in box_data]

    fig, ax = plt.subplots(figsize=(15, 4.5))
    bp = ax.boxplot(box_data, patch_artist=True, showfliers=False,
                     widths=0.7, medianprops={'color': 'black', 'linewidth': 0.8})
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c); patch.set_alpha(0.75)
    ax.scatter(range(1, len(means)+1), means, marker='D', color=C_DIAMOND,
               s=10, zorder=10)

    ax.set_xticks(range(1, len(all_dists)+1))
    ax.set_xticklabels(labels, rotation=50, ha='right', fontsize=6)
    ax.set_ylabel('Cognitive Score (BLEU + ROUGE-L + 0.1·CIDEr)')
    ax.set_xlim(0.3, len(all_dists) + 0.7)
    if uav_count > 0:
        ax.axvline(x=uav_count + 0.5, color='black', linestyle='--', linewidth=0.6, alpha=0.4)
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor=C_UAV, alpha=0.75, label=f'UAV-Specific ({uav_count})'),
        Patch(facecolor=C_GENERIC, alpha=0.75, label=f'Generic ({len(all_dists)-uav_count})'),
    ], loc='upper right', frameon=True)
    save_fig(fig, 'fig1_distribution')


# ── Figure 2: UAV vs Generic Bar Chart ───────────────────────────
def fig2_uav_vs_generic(data):
    print("Figure 2: UAV vs Generic distortion comparison")
    dist_scores = defaultdict(list)
    for item in data:
        info = parse_sid(item['sample_id'])
        dist_scores[info['distortion']].append(cognitive(item))

    all_dists = sorted(dist_scores.keys(),
                       key=lambda d: (0 if d in UAV_DISTS else 1, d))
    labels = [UAV_DISPLAY.get(d, d.replace('_', ' ').title()) for d in all_dists]
    means = [np.mean(dist_scores[d]) for d in all_dists]
    stds  = [np.std(dist_scores[d]) for d in all_dists]
    colors = [C_UAV if d in UAV_DISTS else C_GENERIC for d in all_dists]
    uav_count = sum(1 for d in all_dists if d in UAV_DISTS)

    fig, ax = plt.subplots(figsize=(14, 4))
    xs = range(len(all_dists))
    bars = ax.bar(xs, means, color=colors, alpha=0.8, width=0.7,
                  edgecolor='white', linewidth=0.3)
    ax.errorbar(xs, means, yerr=stds, fmt='none', ecolor='gray',
                capsize=2, linewidth=0.6)
    # Mean reference line
    grand_mean = np.mean(means)
    ax.axhline(y=grand_mean, color='gray', linestyle='--', linewidth=0.7, alpha=0.6)

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=50, ha='right', fontsize=6)
    ax.set_ylabel('Mean Cognitive Score')
    ax.set_xlim(-0.5, len(all_dists) - 0.5)
    if uav_count > 0:
        ax.axvline(x=uav_count - 0.5, color='black', linestyle='--', linewidth=0.6, alpha=0.4)
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor=C_UAV, alpha=0.8, label='UAV-Specific'),
        Patch(facecolor=C_GENERIC, alpha=0.8, label='Generic'),
    ], loc='upper right', frameon=True)
    save_fig(fig, 'fig2_uav_vs_generic')


# ── Figure 3: 3D Score Breakdown (BLEU / ROUGE-L / CIDEr) ───────
def fig3_three_dimensions(data):
    print("Figure 3: 3-D score breakdown per distortion category")
    # Aggregate by category
    cat_bleu  = defaultdict(list)
    cat_rouge = defaultdict(list)
    cat_cider = defaultdict(list)
    for item in data:
        info = parse_sid(item['sample_id'])
        cat = DIST_CATEGORY.get(info['distortion'], 'Other')
        cat_bleu[cat].append(item['bleu'])
        cat_rouge[cat].append(item['rouge_l'])
        cat_cider[cat].append(item['cider'])

    cat_order = ['UAV', 'Blur', 'Luminance', 'Chrominance', 'Noise',
                 'Compression', 'Spatial', 'Other']
    cats = [c for c in cat_order if c in cat_bleu]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, metric_name, metric_data, ylabel in [
        (axes[0], 'BLEU (Precision)', cat_bleu, 'Mean BLEU Score'),
        (axes[1], 'ROUGE-L (Recall)', cat_rouge, 'Mean ROUGE-L Score'),
        (axes[2], 'CIDEr (Semantic)', cat_cider, 'Mean CIDEr Score'),
    ]:
        means = [np.mean(metric_data[c]) for c in cats]
        stds  = [np.std(metric_data[c]) for c in cats]
        bar_colors = [CATEGORY_COLORS.get(c, '#888888') for c in cats]
        bars = ax.bar(range(len(cats)), means, color=bar_colors, alpha=0.85,
                      width=0.65, edgecolor='white', linewidth=0.3)
        ax.errorbar(range(len(cats)), means, yerr=stds, fmt='none',
                     ecolor='gray', capsize=3, linewidth=0.6)
        ax.set_xticks(range(len(cats)))
        ax.set_xticklabels(cats, rotation=30, ha='right', fontsize=7)
        ax.set_ylabel(ylabel)
        ax.set_title(metric_name, fontsize=FONT_SIZE, fontweight='bold')

    plt.tight_layout()
    save_fig(fig, 'fig3_three_dimensions')


# ── Figure 4: JND Sensitivity Classification ─────────────────────
def fig4_jnd_sensitivity(data):
    print("Figure 4: JND-based distortion sensitivity")
    # Group cognitive scores by distortion
    dist_scores = defaultdict(list)
    for item in data:
        info = parse_sid(item['sample_id'])
        dist_scores[info['distortion']].append(cognitive(item))

    all_dists = sorted(dist_scores.keys(),
                       key=lambda d: (0 if d in UAV_DISTS else 1, d))
    n = len(all_dists)

    # For each distortion, compute: mean cognitive score and assign JND tier
    # Sort by mean score ascending (lower = more severe)
    dist_mean = {d: np.mean(dist_scores[d]) for d in all_dists}
    sorted_by_severity = sorted(all_dists, key=lambda d: dist_mean[d])

    # JND tiers: bottom 1/3 = Severe, middle 1/3 = Medium, top 1/3 = Mild
    n_each = n // 3
    jnd = {}
    for i, d in enumerate(sorted_by_severity):
        if i < n_each:
            jnd[d] = 'Severe'
        elif i < 2 * n_each:
            jnd[d] = 'Medium'
        else:
            jnd[d] = 'Mild'

    jnd_colors = {'Mild': '#4CAF50', 'Medium': '#FFC107', 'Severe': '#F44336'}

    # Build heatmap data: distortion × (BLEU contribution, ROUGE-L contribution, CIDEr contribution, Mean Cog)
    metrics = ['BLEU', 'ROUGE-L', 'CIDEr×0.1', 'Cognitive']
    heatmap = np.zeros((n, 4))
    dist_labels = []
    for i, d in enumerate(all_dists):
        scores = dist_scores[d]
        heatmap[i, 0] = np.mean([item['bleu'] for item in data
                                  if parse_sid(item['sample_id'])['distortion'] == d])
        heatmap[i, 1] = np.mean([item['rouge_l'] for item in data
                                  if parse_sid(item['sample_id'])['distortion'] == d])
        heatmap[i, 2] = 0.1 * np.mean([item['cider'] for item in data
                                        if parse_sid(item['sample_id'])['distortion'] == d])
        heatmap[i, 3] = dist_mean[d]
        dist_labels.append(UAV_DISPLAY.get(d, d.replace('_', ' ').title()))

    # Normalize each column to [0,1]
    heatmap_norm = (heatmap - heatmap.min(axis=0)) / (heatmap.max(axis=0) - heatmap.min(axis=0) + 1e-10)

    # Sort by JND tier then by cognitive score
    jnd_order = {'Severe': 0, 'Medium': 1, 'Mild': 2}
    sort_idx = sorted(range(n), key=lambda i: (jnd_order[jnd[all_dists[i]]], dist_mean[all_dists[i]]))
    heatmap_sorted = heatmap_norm[sort_idx]
    labels_sorted = [dist_labels[i] for i in sort_idx]
    jnd_sorted = [jnd[all_dists[i]] for i in sort_idx]

    fig, ax = plt.subplots(figsize=(6, 12))
    im = ax.imshow(heatmap_sorted, aspect='auto', cmap='YlOrRd')

    # JND color bar on left
    for i, tier in enumerate(jnd_sorted):
        ax.axhline(y=i, color=jnd_colors[tier], linewidth=3, alpha=0.6)
        ax.text(-0.5, i, tier[0], ha='center', va='center', fontsize=7,
                fontweight='bold', color=jnd_colors[tier])

    ax.set_yticks(range(n))
    ax.set_yticklabels(labels_sorted, fontsize=6.5)
    ax.set_xticks(range(4))
    ax.set_xticklabels(metrics, rotation=30, ha='right')

    # Add JND legend
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor=jnd_colors['Mild'], alpha=0.6, label='Mild (top 1/3)'),
        Patch(facecolor=jnd_colors['Medium'], alpha=0.6, label='Medium (mid 1/3)'),
        Patch(facecolor=jnd_colors['Severe'], alpha=0.6, label='Severe (bottom 1/3)'),
    ], loc='lower left', frameon=True, fontsize=7,
              bbox_to_anchor=(0, -0.02))

    plt.tight_layout()
    save_fig(fig, 'fig4_jnd_sensitivity')


# ── Figure 5: Per-Intensity Degradation ──────────────────────────
def fig5_intensity_curves(data):
    print("Figure 5: Per-intensity degradation curves")
    # Per distortion × intensity: mean cognitive score
    di_scores = defaultdict(lambda: defaultdict(list))
    for item in data:
        info = parse_sid(item['sample_id'])
        if info['intensity'] is not None:
            di_scores[info['distortion']][info['intensity']].append(cognitive(item))

    intensities = sorted(set(info['intensity'] for item in data
                             if parse_sid(item['sample_id'])['intensity'] is not None))

    # Compute means per distortion × intensity
    dist_intensity_mean = {}
    for d in di_scores:
        dist_intensity_mean[d] = {}
        for lev in intensities:
            if di_scores[d].get(lev):
                dist_intensity_mean[d][lev] = np.mean(di_scores[d][lev])

    # UAV panel (6 subplots)
    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    axes = axes.flatten()
    for idx, d in enumerate(UAV_DISTS):
        ax = axes[idx]
        if d in dist_intensity_mean:
            levs = sorted(dist_intensity_mean[d].keys())
            vals = [dist_intensity_mean[d][l] for l in levs]
            ax.plot(levs, vals, 'o-', color=C_UAV, linewidth=1.5, markersize=5)
            # Fill between
            ax.fill_between(levs, [v - 0.02 for v in vals], [v + 0.02 for v in vals],
                            alpha=0.15, color=C_UAV)
        ax.set_title(UAV_DISPLAY.get(d, d), fontsize=8, fontweight='bold')
        ax.set_xlabel('Distortion Intensity')
        ax.set_ylabel('Mean Cognitive Score')
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.2, linewidth=0.3)

    plt.tight_layout()
    save_fig(fig, 'fig5_intensity_curves')


# ── Figure 6: Source Comparison (Sim vs Real) ────────────────────
def fig6_source_comparison(data):
    print("Figure 6: Source comparison (Sim vs Real)")
    source_scores = defaultdict(list)
    source_dists = defaultdict(lambda: defaultdict(list))
    for item in data:
        info = parse_sid(item['sample_id'])
        src = info['source']
        source_scores[src].append(cognitive(item))
        source_dists[src][info['distortion']].append(cognitive(item))

    # Panel A: Overall score distribution by source
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    sources = sorted(source_scores.keys())
    src_colors = {'Sim3': C_SIM, 'Sim5': C_GENERIC, 'Sim6': C_UAV, 'Real2': C_REAL}
    src_labels = {'Sim3': 'Sim (CARLA-Air)', 'Sim5': 'Sim (AirCop-Bench)',
                  'Sim6': 'Sim (Motion-Scape)', 'Real2': 'Real (MDMT)'}

    # Violin plot
    violin_data = [source_scores[s] for s in sources]
    vp = ax1.violinplot(violin_data, showmeans=True, showmedians=False)
    for i, body in enumerate(vp['bodies']):
        body.set_facecolor(src_colors.get(sources[i], '#888888'))
        body.set_alpha(0.6)
    ax1.set_xticks(range(1, len(sources)+1))
    ax1.set_xticklabels([src_labels.get(s, s) for s in sources], rotation=20, ha='right', fontsize=7)
    ax1.set_ylabel('Cognitive Score')
    ax1.set_title('Score Distribution by Source', fontsize=FONT_SIZE, fontweight='bold')

    # Panel B: UAV distortion scores per source
    uav_means = {}
    for d in UAV_DISTS:
        uav_means[d] = {}
        for s in sources:
            vals = source_dists[s].get(d, [])
            uav_means[d][s] = np.mean(vals) if vals else 0

    x = np.arange(len(sources))
    width = 0.12
    for i, d in enumerate(UAV_DISTS):
        vals = [uav_means[d].get(s, 0) for s in sources]
        bars = ax2.bar(x + i*width, vals, width, label=UAV_DISPLAY.get(d, d),
                       color=CB6[i], alpha=0.8, edgecolor='white', linewidth=0.2)
    ax2.set_xticks(x + width * 2.5)
    ax2.set_xticklabels([src_labels.get(s, s) for s in sources], rotation=20, ha='right', fontsize=7)
    ax2.set_ylabel('Mean Cognitive Score')
    ax2.set_title('UAV Distortion Scores by Source', fontsize=FONT_SIZE, fontweight='bold')
    ax2.legend(fontsize=6, frameon=True, ncol=2)

    plt.tight_layout()
    save_fig(fig, 'fig6_source_comparison')


# ── Summary Statistics & Tables ──────────────────────────────────
def generate_tables(data):
    print("Generating summary statistics...")
    dist_scores = defaultdict(list)
    dist_bleu = defaultdict(list)
    dist_rouge = defaultdict(list)
    dist_cider = defaultdict(list)
    for item in data:
        info = parse_sid(item['sample_id'])
        d = info['distortion']
        cs = cognitive(item)
        dist_scores[d].append(cs)
        dist_bleu[d].append(item['bleu'])
        dist_rouge[d].append(item['rouge_l'])
        dist_cider[d].append(item['cider'])

    all_dists = sorted(dist_scores.keys(), key=lambda d: np.mean(dist_scores[d]))
    print(f"\n{'Distortion':<35s} {'Category':<15s} {'Mean Cog':>8s} {'Std':>8s} {'BLEU':>8s} {'ROUGE':>8s} {'CIDEr':>8s}")
    print("-" * 105)
    for d in all_dists:
        cat = DIST_CATEGORY.get(d, 'Other')
        mc = np.mean(dist_scores[d])
        sc = np.std(dist_scores[d])
        mb = np.mean(dist_bleu[d])
        mr = np.mean(dist_rouge[d])
        ml = np.mean(dist_cider[d])
        marker = " <-- UAV" if d in UAV_DISTS else ""
        print(f"{d:<35s} {cat:<15s} {mc:8.4f} {sc:8.4f} {mb:8.4f} {mr:8.4f} {ml:8.4f}{marker}")

    # Overall stats
    all_cog = [cognitive(item) for item in data]
    print(f"\nOverall: mean={np.mean(all_cog):.4f}, std={np.std(all_cog):.4f}, "
          f"min={np.min(all_cog):.4f}, max={np.max(all_cog):.4f}")

    # UAV vs Generic summary
    uav_cog = [cognitive(item) for item in data
               if parse_sid(item['sample_id'])['distortion'] in UAV_DISTS]
    gen_cog = [cognitive(item) for item in data
               if parse_sid(item['sample_id'])['distortion'] not in UAV_DISTS]
    print(f"\nUAV:     mean={np.mean(uav_cog):.4f}, std={np.std(uav_cog):.4f}, n={len(uav_cog)}")
    print(f"Generic: mean={np.mean(gen_cog):.4f}, std={np.std(gen_cog):.4f}, n={len(gen_cog)}")

    # Per-category summary
    print("\n--- Per Category ---")
    cat_cog = defaultdict(list)
    for item in data:
        cat = DIST_CATEGORY.get(parse_sid(item['sample_id'])['distortion'], 'Other')
        cat_cog[cat].append(cognitive(item))
    for cat in ['UAV', 'Blur', 'Luminance', 'Chrominance', 'Noise', 'Compression', 'Spatial', 'Other']:
        if cat in cat_cog:
            vals = cat_cog[cat]
            print(f"  {cat:<15s}: mean={np.mean(vals):.4f}, std={np.std(vals):.4f}, n={len(vals)}")


# ── Main ─────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 60)
    print("UAV-Embodied-IQA: Dataset Analysis Figures")
    print("=" * 60)

    data = load_data()
    print(f"Loaded {len(data)} annotation entries")

    generate_tables(data)
    fig1_distribution(data)
    fig2_uav_vs_generic(data)
    fig3_three_dimensions(data)
    fig4_jnd_sensitivity(data)
    fig5_intensity_curves(data)
    fig6_source_comparison(data)

    print("\nDone! All figures saved to figures/")
