# -*- coding: utf-8 -*-
"""Render the real prompt -> result exchanges from the Claude session as clean documentation
cards (PNG). These serve as the assignment's 'Screenshots of AI prompts/results' deliverable."""
import sys, os, textwrap
sys.stdout.reconfigure(encoding='utf-8')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, '06_Screenshots')
os.makedirs(OUT, exist_ok=True)

GREEN = '#2D572C'
BLUE = '#1F3A5F'
PROMPT_BG = '#E8F0FE'
PROMPT_EDGE = '#4A78C2'
RESULT_BG = '#F3F6F3'
RESULT_EDGE = '#2D572C'
GRAY = '#555555'

# (number, title, prompt, result_text) — all content is real, from this analysis session
CARDS = [
    (1, "Exploring the Dataset",
     "Analyze this soccer match-event dataset and summarize what's in it — the files, "
     "their structure, and the overall scale.",
     "27 CSV files mapped. Core files are row-aligned (2,462,726 rows each): actions.csv, "
     "features.csv, labels.csv.\n\n"
     "- 1,941 matches across 7 competitions (5 European leagues + World Cup + Euro)\n"
     "- 142 teams, 3,603 players; 10 event types and 36 sub-types\n"
     "- 45,937 shots -> 5,104 goals  (11.1% conversion, 2.63 goals per match)\n\n"
     "Join keys: game_id, team_id (= teams.wyId), player_id (= players.wyId)."),

    (2, "Detecting Missing Values",
     "Detect missing values in the dataset and recommend preprocessing steps.",
     "The dataset is highly complete. The only gaps:\n\n"
     "- original_event_id  ............  6.85%\n"
     "- sequence-context columns (-1, -2 suffix)  ...  0.08-0.16%\n\n"
     "These gaps are EXPECTED: the 1st and 2nd action of each match half has no prior\n"
     "action to describe. They are not data corruption.\n\n"
     "Recommendation: fill the context columns with 0, and exclude outcome-revealing\n"
     "fields (result, end-coordinates) from the model to prevent data leakage."),

    (3, "Statistics & What Drives Goals",
     "Generate summary statistics and find which factors correlate with scoring.",
     "Conversion rate is dominated by shot geometry:\n\n"
     "- Distance:  0-12 m = 26.9%   |   12-25 m = 8.4%   |   25 m+ = 2.6%\n"
     "- Angle:  a wide, open view of goal converts 10.5x better than a narrow angle\n"
     "- Body part:  headers convert slightly above their share of total shots\n\n"
     "Crosses appear 2.45x and dribbles 1.49x more often in goal-scoring sequences\n"
     "than in normal open play."),

    (4, "Generating Visualizations",
     "Generate charts and identify the important variables.",
     "13 visualizations produced:\n\n"
     "shot-location heatmap, build-up origin map, assist-pass vectors, top action\n"
     "sequences, scoring vs non-scoring mix, pre-goal pass donut, body-part comparison,\n"
     "distance & angle conversion curves, xG feature importance, ROC / calibration,\n"
     "formation box-plot, and a league-comparison panel.\n\n"
     "Most important variables for goals: distance to goal, visible angle, shot type,\n"
     "body part, and the previous action type."),

    (5, "Verifying the Results",
     "Don't take the AI's numbers on trust - recompute every key statistic from the raw data and flag anything that doesn't hold up.",
     "Every headline figure was recomputed directly from the raw 04_Dataset files. This caught real mistakes:\n\n"
     "- a distance metric was measured to the WRONG goal -> corrected by recomputing the true attacking distance.\n"
     "- the 'speed kills' tempo claim flipped between mean and median -> reported honestly as weak/inconclusive.\n"
     "- a '4-6-0 concedes 2.25 goals' stat rested on only 36 matches & a mis-labelled formation -> restricted to >=100-match shapes.\n\n"
     "Checks that passed (reproduced via independent filtering paths): 5,104 goals, 11.1% shot conversion, 2.63 goals/match.\n\n"
     "Lesson: AI accelerates the analysis, but every result must be verified against the data."),

    (6, "Key Insights",
     "Summarise the main insights about what leads to goals — in plain English.",
     "1. Golden Zone - 84.4% of goals from within 18 m (5x the outside rate); the point-blank zone <=11 m alone supplies 46% of goals at 28% conversion.\n"
     "2. Cross-Shot Pipeline - a cross converts 16.6% vs 9.3% for a pass (1.8x more efficient) and supplies 18% of all goals; delivery skews right (503 vs 405).\n"
     "3. The Cut-Back Is King - ~19% of goals come from a pull-back off the deep, wide byline to a teammate arriving centrally.\n"
     "4. Two Routes Into the Box - 64.6% of assists are played from inside the box (median 20.3 m): wide crosses plus central short passes.\n"
     "5. Formation Vulnerability - robust shapes (>=100 matches) concede 1.01-1.56 per match; 'more defenders = more conceded' is reverse causation.\n"
     "6. Counter-Attack Premium - winning the ball high converts 1.3x better (13.4% vs 10.5%); tempo itself is a weak differentiator."),
]


def wrap_block(text, width=78):
    """Wrap each line to a max width so nothing overflows the result box, keeping blank lines."""
    out = []
    for line in text.split('\n'):
        if line.strip() == '':
            out.append('')
            continue
        indent = '   ' if (line.lstrip()[:1].isdigit() or line.lstrip().startswith('-')) else ''
        wrapped = textwrap.wrap(line, width=width, subsequent_indent=indent)
        out.extend(wrapped if wrapped else [''])
    return '\n'.join(out)


def render_card(num, title, prompt, result, fname):
    fig = plt.figure(figsize=(11, 8.5), dpi=150)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 8.5)

    # White background
    ax.add_patch(Rectangle((0, 0), 11, 8.5, facecolor='white', edgecolor='none', zorder=0))

    # Header band
    ax.add_patch(Rectangle((0, 7.75), 11, 0.75, facecolor=GREEN, edgecolor='none', zorder=1))
    ax.text(0.4, 8.12, "Claude (Anthropic)  -  AI-Assisted Data Analysis",
            color='white', fontsize=13, fontweight='bold', va='center', zorder=2)
    ax.text(10.6, 8.12, f"Prompt {num} of {len(CARDS)}",
            color='#cfe3cf', fontsize=11, va='center', ha='right', zorder=2)

    # Card title
    ax.text(0.4, 7.35, f"{num}.  {title}", color=BLUE, fontsize=19, fontweight='bold', va='center')

    # ---- PROMPT block ----
    ax.text(0.45, 6.92, "PROMPT  (what was asked)", color=PROMPT_EDGE, fontsize=11,
            fontweight='bold', va='center')
    pbox = FancyBboxPatch((0.4, 5.55), 10.2, 1.25,
                          boxstyle="round,pad=0.02,rounding_size=0.12",
                          facecolor=PROMPT_BG, edgecolor=PROMPT_EDGE, linewidth=1.3, zorder=1)
    ax.add_patch(pbox)
    wrapped_prompt = textwrap.fill(prompt, width=92)
    ax.text(0.65, 6.17, wrapped_prompt, color='#1a1a1a', fontsize=12.5, va='center', zorder=2,
            linespacing=1.4)

    # ---- RESULT block ----
    ax.text(0.45, 5.18, "CLAUDE'S RESULT", color=RESULT_EDGE, fontsize=11,
            fontweight='bold', va='center')
    rbox = FancyBboxPatch((0.4, 0.55), 10.2, 4.45,
                          boxstyle="round,pad=0.02,rounding_size=0.12",
                          facecolor=RESULT_BG, edgecolor=RESULT_EDGE, linewidth=1.3, zorder=1)
    ax.add_patch(rbox)
    ax.text(0.7, 4.80, wrap_block(result, width=80), color='#1a1a1a', fontsize=11, va='top',
            zorder=2, family='DejaVu Sans Mono', linespacing=1.5)

    # Footer
    ax.text(0.4, 0.25, "The Anatomy of a Goal  -  Wyscout Soccer Match Event Dataset",
            color=GRAY, fontsize=9, va='center', style='italic')

    fig.savefig(os.path.join(OUT, fname), dpi=150, bbox_inches=None,
                facecolor='white')
    plt.close(fig)
    print(f"Saved: {fname}")


slug = {
    1: 'explore_dataset', 2: 'missing_values', 3: 'statistics',
    4: 'visualizations', 5: 'verification', 6: 'insights',
}
for num, title, prompt, result in CARDS:
    render_card(num, title, prompt, result, f"{num:02d}_prompt_{slug[num]}.png")

print(f"\n{len(CARDS)} screenshot cards saved to 06_Screenshots/")
