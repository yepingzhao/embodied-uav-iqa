"""Shared plotting style for UAV-Embodied-IQA paper figures."""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import json
import os
from pathlib import Path

# ── Style constants ──────────────────────────────────────────────
FONT_SIZE = 10
DPI = 300
FORMAT = "pdf"
FIG_DIR = Path(__file__).resolve().parent  # figures/
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

matplotlib.rcParams.update({
    'font.size': FONT_SIZE,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'axes.labelsize': FONT_SIZE,
    'axes.titlesize': FONT_SIZE,
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
    'mathtext.fontset': 'stix',
})

# Colorblind-safe palette (IBM/Okabe-Ito adapted)
COLORS = {
    'uav':      '#E69F00',  # orange — UAV distortions (highlight)
    'uav_light':'#F0C060',
    'generic':  '#56B4E9',  # sky blue — generic distortions
    'grid':     '#E0E0E0',  # light grid
    'accent1':  '#009E73',  # green
    'accent2':  '#CC79A7',  # purple
    'accent3':  '#D55E00',  # vermillion
}
CB_PALETTE = ['#E69F00', '#56B4E9', '#009E73', '#CC79A7', '#D55E00', '#0072B2', '#F0E442']

# 6 UAV-specific distortion types
UAV_DISTORTIONS = [
    'propeller_vibration_blur',
    'atmospheric_scattering_haze',
    'six_dof_viewpoint_blur',
    'communication_packet_loss',
    'low_res_super_resolution',
    'propeller_shadow',
]

UAV_DISPLAY_NAMES = {
    'propeller_vibration_blur':     'Propeller\nVibration',
    'atmospheric_scattering_haze':  'Atmospheric\nScattering',
    'six_dof_viewpoint_blur':       '6DoF Viewpoint\nBlur',
    'communication_packet_loss':    'Packet-Loss\nBlocks',
    'low_res_super_resolution':     'Low-Res\n+SR',
    'propeller_shadow':             'Propeller\nShadow',
}

DISTORTION_CATEGORIES = {
    'propeller_vibration_blur':     'UAV',
    'atmospheric_scattering_haze':  'UAV',
    'six_dof_viewpoint_blur':       'UAV',
    'communication_packet_loss':    'UAV',
    'low_res_super_resolution':     'UAV',
    'propeller_shadow':             'UAV',
    'gaussian_blur':                'Blur',
    'lens_blur':                    'Blur',
    'motion_blur':                  'Blur',
    'brighten_max':                 'Luminance',
    'brighten_avg':                 'Luminance',
    'darken_max':                   'Luminance',
    'darken_min':                   'Luminance',
    'darken_avg':                   'Luminance',
    'color_diffusion':              'Chrominance',
    'color_shift':                  'Chrominance',
    'color_quantize':               'Chrominance',
    'white_noise':                  'Noise',
    'color_noise':                  'Noise',
    'impulse_noise':                'Noise',
    'multiplicative_noise':         'Noise',
    'gaussian_denoise':             'Noise',
    'cnn_denoise':                  'Noise',
    'jpeg_compression':             'Compression',
    'jp2k_compression':             'Compression',
    'webp_compression':             'Compression',
    'spatial_warp':                 'Spatial',
    'spatial_scale':                'Spatial',
    'clock_jittering':              'Spatial',
    'resolution_limit':             'Spatial',
    'grayscale':                    'Other',
    'sharpness':                    'Other',
    'contrast':                     'Other',
    'block_lost':                   'Other',
    'block_interpolation':          'Other',
    'block_exchange':               'Other',
}


def save_fig(fig, name, fmt=FORMAT):
    """Save figure to FIG_DIR with consistent naming."""
    path = FIG_DIR / f"{name}.{fmt}"
    fig.savefig(str(path))
    print(f"Saved: {path}")
    plt.close(fig)


def load_all_annotations(data_dir=None):
    """Load all VLM annotation data and return as list of dicts."""
    if data_dir is None:
        data_dir = DATA_DIR / "annotated" / "vlm" / "Qwen2-VL"
    data_dir = Path(data_dir)
    all_data = []
    for split in ['train', 'test']:
        split_dir = data_dir / split
        if split_dir.exists():
            for fname in sorted(split_dir.glob('*.json')):
                with open(fname) as f:
                    data = json.load(f)
                all_data.extend(data)
    return all_data


def compute_cognitive_score(item):
    """Cognitive score = BLEU + ROUGE-L + 0.1*CIDEr."""
    return item['bleu'] + item['rouge_l'] + 0.1 * item['cider']


def parse_sample_id(sid):
    """Parse sample_id into components.
    Format: {source}__{split}__{scene_frame}__{task}__{distortion}_L{intensity}
    """
    parts = sid.split('__')
    source = parts[0]
    split = parts[1]
    scene_frame = parts[2] if len(parts) > 2 else ''
    task = parts[3] if len(parts) > 3 else ''
    dist_str = parts[4] if len(parts) > 4 else ''

    # Parse distortion name and intensity
    dist_name = dist_str
    intensity = None
    if '_L' in dist_str:
        idx = dist_str.rfind('_L')
        dist_name = dist_str[:idx]
        try:
            intensity = int(dist_str[idx+2:]) / 10.0  # L08 → 0.8
        except ValueError:
            intensity = None

    return {
        'source': source,
        'split': split,
        'task': task,
        'distortion': dist_name,
        'intensity': intensity,
    }
