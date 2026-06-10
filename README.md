# The Anatomy of a Goal — AI-Assisted Data Analysis

**Assignment:** AI-Assisted Data Analysis Using Agentic AI
**Author:** Abdallah Mattour
**AI Tool Used:** Claude (Anthropic)
**Dataset:** [Wyscout Soccer Match Event Dataset (Kaggle)](https://www.kaggle.com/datasets/aleespinosa/soccer-match-event-dataset)

A full investigation of goal-scoring patterns across Europe's top leagues + the World Cup and Euro — what action sequences, pass types, pitch zones, distances/angles, and formations produce goals — including a machine-learning Expected Goals (xG) model.

---

## 📁 Folder Structure

| Folder | Contents | What's inside |
|--------|----------|---------------|
| **01_Report/** | The written report | `Manhaji_Report_Anatomy_of_a_Goal.docx` and `.pdf` — the 5-page analytical report covering all four assignment tasks |
| **02_Notebook/** | The analysis notebook | `soccer_goal_analysis.ipynb` — 31 cells, 13 charts, full code + outputs + the xG model |
| **03_Figures/** | The visualizations | 13 chart images (PNG) extracted from the notebook — usable as screenshots of results |
| **04_Dataset/** | The data | 27 CSV files (the raw Wyscout dataset) |
| **05_Scripts/** | Reproducible build scripts | The Python scripts that generate and execute the notebook, extract the figures, and build the report |
| **06_Screenshots/** | AI prompts & results | 6 cards documenting the real prompts given to Claude and the results it produced (assignment deliverable #3) |

---

## ▶️ How to Reproduce

All scripts auto-detect the project root, so they work from anywhere as long as the folder structure above is intact.

```bash
cd 05_Scripts

# 1. Generate the notebook (.ipynb) from source
python generate_notebook.py        # writes 02_Notebook/soccer_goal_analysis.ipynb

# 2. Execute it (runs all cells, ~2-3 min, needs 04_Dataset present)
python execute_notebook.py

# 3. Extract the 13 charts as PNGs
python extract_charts.py           # writes 03_Figures/*.png

# 4. Build the Word report (requires python-docx; PDF needs MS Word/LibreOffice)
python generate_report.py          # writes 01_Report/Manhaji_Report_Anatomy_of_a_Goal.docx
```

**Requirements:** Python 3, `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `nbformat`, `nbclient`, `python-docx`.

---

## 📊 The Analysis at a Glance

- **Scale:** 2,462,726 on-ball actions · 1,941 matches · 7 competitions · 5,104 goals (11.1% shot conversion)
- **xG model:** Gradient Boosting, ROC-AUC **0.908**, well-calibrated
- **Six key findings:** the Golden Zone, the Cross-Shot Pipeline, a falsified "speed kills" hypothesis, the xG Sweet Spot, Formation Vulnerability, and the Counter-Attack Premium

> The report (01_Report) is the main deliverable to read first. The notebook (02_Notebook) contains the full reproducible analysis. The figures (03_Figures) are the visual results.
