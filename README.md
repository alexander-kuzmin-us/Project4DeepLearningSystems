# Project 4: Deep Learning Systems

Land-use classification of EuroSAT Sentinel-2 satellite images with a CNN in PyTorch, plus one controlled experiment: **baseline (no dropout) vs. the identical model with dropout p = 0.5**, each trained with 3 seeds.

## Submission files

| File | What it is |
|---|---|
| `deep_learning.ipynb` | Full experiment, executed top-to-bottom with outputs |
| `module_summary.pdf` | Deep Learning Systems Analysis Report (APA 7) |
| `requirements.txt` | `pip freeze` of the project environment |
| `data/README.md` | Dataset source, licence, citation and download instructions |

## Reproduce

Create and activate a virtual environment, then install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Run the notebook top-to-bottom (or open it in Jupyter / VS Code with the `.venv` kernel):

```bash
jupyter nbconvert --to notebook --execute --inplace deep_learning.ipynb --ExecutePreprocessor.timeout=-1
```

Run `deactivate` to leave the environment when you're done.

The notebook downloads EuroSAT (~90 MB) on first run. On an Apple M4 (MPS backend) the full run takes about 35 minutes; it also runs on CUDA or CPU (much slower).

## Other files

- `build_notebook.py`: source of the notebook (regenerates an unexecuted `deep_learning.ipynb`).
- `outputs/figures/`, `outputs/metrics/`: figures and metric CSVs written by the notebook and used by the report.
- `report/build_report.py`, `report/apa_docx.py`: write the report into the provided APA template (`report/module_summary.docx`); needs `python-docx`.
- `IMPLEMENTATION_PLAN.md`: the plan and reasoning behind each design decision.
