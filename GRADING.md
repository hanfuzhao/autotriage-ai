# Grading map: where every rubric item lives

This file mirrors the project checklist item by item so nothing has to be hunted for.
Section numbers refer to `TECHNICAL_REPORT.pdf` (same content as `TECHNICAL_REPORT.md`).

- **Live app:** https://autotriage-ai-unfvnsiy6a-uc.a.run.app
- **Repo:** https://github.com/hanfuzhao/autotriage-ai
- **Written report:** `TECHNICAL_REPORT.pdf` in this repo (13 pages, figures embedded)

---

## Project Topic & Originality

| Checklist item | Where |
|---|---|
| Project topic clearly defined and relevant to the module (NLP) | Report section 1, Problem Statement. Task is 14-class text classification of complaint narratives |
| Project is new work, not reused from another course, research, or job | Report section 3, Originality statement, first paragraph states this explicitly |
| Domain choice shows real-world relevance or creativity | Report section 1 and section 9. Owner-facing vehicle safety triage, built on real NHTSA records |
| Originality statement included | Report section 3, "Originality statement", four numbered points contrasting the named prior work |

## Modeling Requirements

| Checklist item | Where |
|---|---|
| Naive baseline implemented | `scripts/model.py::NaiveBaseline`, artifact `models/naive.pkl` |
| Classical (non-deep) ML model implemented | `scripts/model.py::ClassicalModel`, TF-IDF + Logistic Regression, artifact `models/classical.pkl` |
| Neural network / deep learning model implemented | `scripts/model.py::TextCNNModel`, TextCNN + GloVe transfer learning, artifact `models/deep_textcnn.pt` |
| All three models present in the repository | `models/` contains `naive.pkl`, `classical.pkl`, `deep_textcnn.pt`, plus `labels.json` |
| Clear documentation explaining where each model lives | README, "The three required models" table, and this file |
| Deployed model | TextCNN. `GET /health` on the live app reports `"deployed":"deep"` and `"models_loaded":3` |

## Experimentation & Analysis

| Checklist item | Where |
|---|---|
| At least one focused experiment conducted | Report section 7.1, training-set-size sensitivity. Code in `scripts/experiment.py::training_size_experiment` |
| Experiment is well-motivated | Report section 7.1, "Motivation" paragraph, the cold-start question |
| Experimental setup clearly described | Report section 7.1, "Experimental setup": independent and dependent variables, procedure, sampling, controls held constant, validation use, model budget, metric choice, and a stated limitation |
| Results meaningfully interpreted | Report section 7.1, "Interpretation", plus sections 7.2 to 7.4 |
| Experiment directly informs modeling or system design | Report section 7.1, "Recommendation", and section 7.3, which sets the abstention policy used in the app |
| Supporting experiments | 7.2 noise robustness, 7.3 confidence-gated abstention, 7.4 head vs tail |

## Interactive Application

| Checklist item | Where |
|---|---|
| Publicly accessible URL provided | https://autotriage-ai-unfvnsiy6a-uc.a.run.app |
| Application live for 1 week after submission | Google Cloud Run, plus a keep-alive GitHub Action pinging `/health` |
| App runs inference only, no training in production | `main.py` loads the three saved models at startup and only calls `predict`. No training code path is reachable from any route |
| Application runs successfully when graded | `GET /health` returns status ok with 3 models loaded |
| Model inference works end-to-end | `POST /api/predict` returns prediction, confidence, top-3, word-level explanation, safety tier |
| Interface usable and thoughtfully designed | `templates/index.html`, `static/css/style.css`. Filter-free single input, example chips, confidence ring, highlighted evidence |
| UX is a product-level proof of concept | Safety tier with a plain next step, three-model comparison panel, explanations, not a bare form |
| App stability verified | Verified over HTTP end to end, no broken links or missing assets |

## Written Report

All sections are in `TECHNICAL_REPORT.pdf`.

| Checklist item | Section |
|---|---|
| Problem Statement | 1 |
| Data Sources | 2 |
| Related Work (literature review) | 3, names the specific prior studies on this database, Ghazizadeh, McDonald and Lee (2014), Ghazizadeh and Lee (2012), and others |
| Evaluation Strategy & Metrics | 4 |
| Metrics justified | 4, why macro-F1 over accuracy given the long tail |
| Modeling Approach | 5 |
| Data Processing Pipeline + rationale for each step | 5.1, seven numbered steps, each with its reason |
| Hyperparameter Tuning Strategy | 5.3 strategy, 5.4 the search that was actually run, with the grid, protocol, and results table. Code: `scripts/tune.py`. Raw output: `data/outputs/tuning.json` |
| Model Evaluations, all 3 | 6.1, all three scored on the held-out test split only, n = 5,602 |
| Quantitative comparison across models | 6.1 table, accuracy / macro-F1 / weighted-F1 |
| Visualizations included | 6 figures embedded in the PDF, all 8 in `data/outputs/plots/` |
| Confusion matrices | 6.3, `confusion_deep.png`, plus `confusion_classical.png` and `confusion_naive.png` |
| At least 5 specific mispredictions identified | 8, exactly five, each quoting the actual complaint text |
| Root causes explained | 8, a root cause per case, plus the cross-cutting theme |
| Concrete mitigation strategies proposed | 8, a mitigation per case |
| Experimental plan | 7.1 |
| Results reported | 6 and 7 |
| Interpretation provided | 6.1 to 6.3, 7.1 to 7.4 |
| Actionable recommendations | 7.1 and 7.3 |
| Conclusions | 11 |
| Future Work, another semester | 11 |
| Commercial Viability Statement | 9 |
| Ethics Statement | 10 |
| Data splitting stated | 2, 70/15/15 stratified, 26,138 train / 5,602 validation / 5,602 test |

## In-Class Pitch

The pitch is delivered as a video and submitted as a separate link, so the slide deck and
speaking notes are not part of this repository.

| Checklist item | Where |
|---|---|
| Problem & motivation clearly articulated | Video, opening section |
| Approach overview concise and accurate | Video, followed by the originality contrast |
| Live demo shown, or video link | Video, demonstrated against the live URL above |
| Results, insights, key findings highlighted | Video, the model comparison and the cold-start finding |
| Presentation respects the hard time limit | Delivered inside the 5:00 limit |

## Code & Repository, Git Best Practices

| Checklist item | Where |
|---|---|
| Git best practices, branches used | Feature branches: `feature/data-and-models`, `feature/webapp`, `feature/glove-and-results`, plus later fix branches |
| PRs made into the main repository | 6 merged pull requests, all through branches, none pushed straight to main |
| Good PR reviews before merging | Each PR carries a review comment stating what was checked before merge |

## Code Quality & Practices

| Checklist item | Where |
|---|---|
| No executable code outside functions or `if __name__ == "__main__"` | All scripts guard their entry points. `main.py` only creates the Flask app and loads models at import, which is required for a WSGI server |
| Code modularized into functions and classes | `scripts/model.py` three model classes with a shared interface, plus `data.py`, `evaluate.py`, `experiment.py`, `tune.py`, `eda.py` |
| Jupyter notebooks only in `notebooks/` | `notebooks/` exists and is empty. No notebooks anywhere else |
| External code and AI usage attributed | Note at the top of `main.py` and `setup.py`, plus the Acknowledgments section in the README. Data and paper citations in the file headers |

## Reproducibility and Documentation

| Checklist item | Where |
|---|---|
| Code is well organized | Repo layout follows the required structure |
| All necessary files to run the project are included | `requirements.txt`, `requirements-deploy.txt`, `Dockerfile`, `Makefile`, committed models |
| Able to run the project with the provided instructions | README Quick start: `make install`, `make data`, `make train`, `make app`. `make fast` gives a seconds-long smoke test |
| Code well commented, docstrings used | Every module and public function carries a docstring |
| Code is readable | Descriptive names, small functions |
| Descriptive README | README covers what it does, the three models, results, quick start, deployment, structure, originality, citations |

## Reproduce everything

```bash
make install
make data          # download the NHTSA corpus
make train         # splits, all three models, experiments, metrics, plots
python -m scripts.tune   # hyperparameter search
make app           # serve at http://localhost:5000
```
