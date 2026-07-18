# Where each requirement lives

A quick map from the Module 2 rubric to the files in this repo.

## The three models

| Requirement | Code | Trained file |
|---|---|---|
| Naive baseline | `scripts/model.py::NaiveBaseline` | `models/naive.pkl` |
| Classical, non deep | `scripts/model.py::ClassicalModel`, TF-IDF and LogReg | `models/classical.pkl` |
| Deep learning | `scripts/model.py::TextCNNModel`, TextCNN | `models/deep_textcnn.pt` |
| Deployed model | TextCNN, served in `main.py` | committed, loads locally |

These are also listed in the README under "The three required models".

## Experiments

The main experiment is training-set-size sensitivity, the learning curves. It is in
`scripts/experiment.py::training_size_experiment`, the numbers are in
`data/outputs/metrics.json` under `experiments.training_size`, the figure is
`data/outputs/plots/learning_curve.png`, and the write-up is section 7 of the report.

Three supporting studies sit alongside it: character-noise robustness
(`robustness_experiment`), confidence-gated abstention (`confidence_gating`), and a
head vs tail per-class breakdown (`head_tail_analysis`).

## The app

`main.py` is a Flask app that runs inference only and loads all three models at startup.
The interface is `templates/index.html` with `static/css/style.css` and
`static/js/app.js`. It shows a prediction with confidence, the top 3, a word-level
explanation, a safety-tier triage banner, a live three-model comparison, and a set of
example complaints. Endpoints are `GET /health`, `GET /api/examples`, and
`POST /api/predict`. It is deployed on Google Cloud Run from the `Dockerfile`, and a
GitHub Action in `.github/workflows/keep-alive.yml` keeps it awake.

## The report

`TECHNICAL_REPORT.md` covers all of the required sections: problem statement, data
sources, related work, evaluation strategy and metrics, modeling approach with the
pipeline rationale and hyperparameter tuning, results with tables and confusion matrices,
an error analysis of five mispredictions with causes and fixes, the experiment write-up,
conclusions, future work, commercial viability, and ethics.

## Code quality and git

All logic sits in functions and classes, with nothing running at the top level outside an
`if __name__ == "__main__"` guard. Data source and paper citations are noted in the file
headers. Notebooks would go under `notebooks/`. The repo was built on feature branches
merged into `main` through pull requests, so the branch structure and PR are visible on
GitHub.

## Reproduce everything

```bash
make install && make data && make train && make app
```
