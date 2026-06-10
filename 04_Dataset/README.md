# Dataset — Wyscout Soccer Match Event Dataset

The raw data is **not stored in this repository** because the files exceed GitHub's
size limits (`features.csv` is 2.2 GB and `actions.csv` is 282 MB; GitHub's hard limit
is 100 MB per file).

## How to get the data
1. Download it from Kaggle:
   https://www.kaggle.com/datasets/aleespinosa/soccer-match-event-dataset
2. Unzip and place all 27 CSV files directly in this `04_Dataset/` folder.

The scripts in `05_Scripts/` and the notebook in `02_Notebook/` read the CSVs from here.

> Note: the notebook (`02_Notebook/soccer_goal_analysis.ipynb`) already contains all
> executed outputs and charts, so it can be read and reviewed **without** downloading the
> data or re-running it. The dataset is only needed to reproduce the analysis from scratch.
