# -*- coding: utf-8 -*-
"""Extract all charts from the assignment notebook as PNGs, auto-named by chart number/title."""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import nbformat, base64, os, re

# Project root = the folder that holds 01_Report, 02_Notebook, ... (parent of 05_Scripts)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
nb_path = os.path.join(ROOT, '02_Notebook', 'soccer_goal_analysis.ipynb')
out_dir = os.path.join(ROOT, '03_Figures')
os.makedirs(out_dir, exist_ok=True)

# friendly slugs keyed by chart number (full 13-chart unified notebook)
slugs = {
    1: 'shot_location_heatmap',
    2: 'sequence_origin_heatmap',
    3: 'assist_pass_vectors',
    4: 'top_action_sequences',
    5: 'scoring_vs_nonscoring',
    6: 'pregoal_pass_donut',
    7: 'bodypart_comparison',
    8: 'distance_vs_conversion',
    9: 'angle_vs_conversion',
    10: 'xg_feature_importance',
    11: 'roc_calibration',
    12: 'formation_boxplot',
}

# wipe old charts first
for f in os.listdir(out_dir):
    if f.endswith('.png'):
        os.remove(os.path.join(out_dir, f))

nb = nbformat.read(nb_path, as_version=4)
count = 0
for cell in nb.cells:
    if cell.cell_type != 'code':
        continue
    # match "CHART N:" or the unnumbered "BONUS CHART:" (league comparison -> 13)
    m = re.search(r'CHART (\d+):', cell.source)
    if m:
        num = int(m.group(1))
        slug = slugs.get(num, f'chart{num}')
    elif 'BONUS CHART:' in cell.source:
        num = 13
        slug = 'league_comparison'
    else:
        continue
    for o in cell.outputs:
        if 'data' in o and 'image/png' in o.get('data', {}):
            name = f'{num:02d}_{slug}.png'
            with open(os.path.join(out_dir, name), 'wb') as imgf:
                imgf.write(base64.b64decode(o['data']['image/png']))
            count += 1
            print(f"Saved: {name}")
            break

print(f"\nTotal {count} charts extracted.")
