import os
import nbformat
from nbclient import NotebookClient
import matplotlib
matplotlib.use('Agg')

# Project root = the folder that holds 01_Report, 02_Notebook, ... (parent of 05_Scripts)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
nb_path = os.path.join(ROOT, '02_Notebook', 'soccer_goal_analysis.ipynb')

print("Loading notebook...")
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

print(f"Executing {len(nb.cells)} cells...")
client = NotebookClient(nb, timeout=600, kernel_name='python3')
client.execute()

print("Saving executed notebook...")
with open(nb_path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print("Done! Notebook executed and saved with all outputs.")
