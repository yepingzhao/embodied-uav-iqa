"""Generate Table 1: Database statistics LaTeX table.

Reports reference image counts, distorted pairs, VLM annotation coverage,
and task distribution for the UAV-Embodied-IQA benchmark.
"""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))

import json
from pathlib import Path
from collections import Counter, defaultdict

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FIG_DIR  = Path(__file__).resolve().parent

# ── Load all splits (train + test) ───────────────────────────────
all_data = []
for split in ['train', 'test']:
    split_dir = DATA_DIR / "annotated" / split
    if split_dir.exists():
        for f in sorted(split_dir.glob("*.json")):
            with open(f, encoding='utf-8') as fp:
                all_data.extend(json.load(fp))

print('Total entries loaded: %d' % len(all_data))

# ── Compute stats ─────────────────────────────────────────────────
ds_counts  = Counter(e['dataset'] for e in all_data)
dist_count = len(set(e['distortion_info']['type'] for e in all_data))
n_uav      = sum(1 for e in all_data if e['distortion_info']['category'] == 'uav')

ds_refs = defaultdict(set)
for e in all_data:
    ds_refs[e['dataset']].add(e['sequence_frame'])

# ── Build LaTeX table ─────────────────────────────────────────────
DS_META = {
    'Sim3':  ('AirCopBench (Sim)',  'Sim'),
    'Sim5':  ('CARLA-Air (Sim)',    'Sim'),
    'Sim6':  ('MotionScape (Sim)',  'Sim'),
    'Real2': ('MDMT (Real)',        'Real'),
}

lines = [
    r'\begin{table}[t]',
    r'\centering',
    r'\small',
    r'\setlength{\tabcolsep}{4pt}',
    (r'\caption{UAV-Embodied-IQA database statistics. '
     r'All splits (train + test). '
     r'VLM annotations from 4 models: Qwen2.5-VL, Qwen2-VL, InternVL2, Mini-InternVL.}'),
    r'\label{tab:db_stats}',
    r'\begin{tabular}{llccc}',
    r'\toprule',
    r'\textbf{Source} & \textbf{Type} & \textbf{Ref.\ Images} '
    r'& \textbf{Dist.\ Types} & \textbf{Annotated Pairs} \\',
    r'\midrule',
]

total_refs = 0; total_pairs = 0
for ds in ['Sim3', 'Sim5', 'Sim6', 'Real2']:
    label, dtype = DS_META.get(ds, (ds, ''))
    n_r = len(ds_refs.get(ds, set()))
    n_p = ds_counts.get(ds, 0)
    lines.append(r'%s & %s & %d & 36 & %d \\' % (label, dtype, n_r, n_p))
    total_refs += n_r; total_pairs += n_p

lines += [
    r'\midrule',
    r'\textbf{Total} & --- & \textbf{%d} & \textbf{36} & \textbf{%d} \\' % (total_refs, total_pairs),
    r'\bottomrule',
    r'\end{tabular}',
    r'\end{table}',
]

table_tex = '\n'.join(lines)
out_path = FIG_DIR / 'TABLE_1_database_stats.tex'
with open(out_path, 'w', encoding='utf-8') as f:
    f.write(table_tex + '\n')

print('Saved: %s' % out_path)
print()
print('--- Key stats ---')
print('Total entries: %d' % len(all_data))
print('Distortion types: %d' % dist_count)
print('UAV pairs: %d (%.1f%%)' % (n_uav, 100 * n_uav / len(all_data)))
print('Per-dataset refs:', {k: len(v) for k, v in ds_refs.items()})
print()
print(table_tex)
