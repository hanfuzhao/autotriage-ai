# Rubric → where to find it

A map from each Module 2 Project requirement to where it lives in this repo.

## Required modeling approaches (all three implemented)

| Requirement | Code | Trained artifact |
|---|---|---|
| Naive baseline | `scripts/model.py::NaiveBaseline` | `models/naive.pkl` |
| Classical (non-DL) ML | `scripts/model.py::ClassicalModel` (TF-IDF + LogReg) | `models/classical.pkl` |
| Deep learning model | `scripts/model.py::TextCNNModel` (TextCNN) | `models/deep_textcnn.pt` |
| **Deployed** model | TextCNN (see `main.py`) | committed, loads locally |

Locations are also documented in `README.md` → "The three required models".

## Required experimentation

- **Primary experiment — training-set-size sensitivity (learning curves):**
  `scripts/experiment.py::training_size_experiment`; results in
  `data/outputs/metrics.json → experiments.training_size`; plot
  `data/outputs/plots/learning_curve.png`; written up in `TECHNICAL_REPORT.md §7`.
- Supporting analyses: character-noise **robustness**
  (`robustness_experiment`), **confidence-gated abstention**
  (`confidence_gating`), and **head/tail** per-class analysis (`head_tail_analysis`).

## Interactive application

- `main.py` (Flask) — inference only, three models loaded at startup.
- UI: `templates/index.html`, `static/css/style.css`, `static/js/app.js`.
- Features: prediction + confidence, top-3, **word-level explanation**, safety-tier
  **triage banner**, live **three-model comparison**, curated examples.
- Endpoints: `GET /health`, `GET /api/examples`, `POST /api/predict`.
- Deploy: `Dockerfile` (CPU torch) → Hugging Face Space; `README.md` front-matter is
  Space config; `.github/workflows/keep-alive.yml` keeps it awake.

## Written report (all sections)

`TECHNICAL_REPORT.md`: Problem Statement · Data Sources · Related Work · Evaluation
Strategy & Metrics · Modeling Approach (pipeline + rationale, hyperparameter tuning,
models evaluated) · Results (quantitative comparison, confusion matrices) · Error
Analysis (5 mispredictions + root causes + mitigations) · Experiment Write-Up (plan,
results, interpretation, recommendations) · Conclusions · Future Work · Commercial
Viability · Ethics.

## Code quality / git

- All logic is in functions/classes; no loose top-level execution (guarded by
  `if __name__ == "__main__"`).
- External references attributed in file headers (NHTSA API; Kim 2014 TextCNN).
- Notebooks only under `notebooks/` (none required; exploration space).
- Git: feature branches → reviewed PRs into `main` (see the PR template in
  `.github/pull_request_template.md` and the repository's Pull Requests tab).

## Reproduce everything

```bash
make install && make data && make train && make app
```
