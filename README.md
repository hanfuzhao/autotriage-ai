---
title: AutoTriage AI
emoji: 🚗
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# AutoTriage AI — Vehicle Complaint Component Classifier

> Module 2 Project · Natural Language Processing · Real NHTSA complaint narratives

Owners describe car problems in plain language — *"the brakes went to the floor"*,
*"phantom braking on the highway"*, *"the airbag light stays on"*. **AutoTriage AI**
reads that free-text narrative and predicts **which vehicle system** the complaint is
about, so a safety team can route thousands of incoming reports to the right engineers
instead of triaging them by hand.

- **Live demo:** https://autotriage-ai-unfvnsiy6a-uc.a.run.app  *(Google Cloud Run)*
- **Task:** single-label text classification over **14 vehicle-component classes**
- **Data:** public **NHTSA ODI Complaints API** (real owner narratives, public domain)

## What it does

Paste a complaint. The app returns the predicted component, a confidence score, the
top-3 candidate systems, a **word-level explanation** (which tokens drove the call), a
**triage banner** (critical / high / moderate safety tier + routing suggestion), and a
side-by-side snapshot of how all three required models score the same text.

## The three required models

| Approach | Where | Trained artifact | What it is |
|---|---|---|---|
| **Naive baseline** | `scripts/model.py::NaiveBaseline` | `models/naive.pkl` | Majority-class predictor — the accuracy floor |
| **Classical ML** | `scripts/model.py::ClassicalModel` | `models/classical.pkl` | TF-IDF (1–2 grams) + class-weighted Logistic Regression |
| **Deep learning** (deployed) | `scripts/model.py::TextCNNModel` | `models/deep_textcnn.pt` | TextCNN (Kim, 2014) + **GloVe transfer learning** (Pennington, 2014) |

All three are implemented, trained, and evaluated. The **TextCNN** is the deployed
model; it is small enough (a few MB) to commit, so the app is fully self-contained and
needs no external model hosting.

## Headline results (held-out test set)

| Model | Accuracy | **Macro-F1** | Weighted-F1 |
|---|---|---|---|
| Naive (majority class) | 0.094 | **0.012** | 0.016 |
| Classical (TF-IDF + LogReg) | 0.774 | **0.768** | 0.775 |
| Deep (TextCNN + GloVe) — *deployed* | 0.771 | **0.762** | 0.770 |

Because the component distribution is long-tailed, **macro-F1** (every safety system
weighted equally) is the headline metric, not accuracy — note the naive baseline scores
0.094 accuracy but only 0.012 macro-F1. GloVe transfer learning lifts the neural model
to parity with the strong classical baseline and closes its cold-start gap (see
`TECHNICAL_REPORT.md §7`). Full numbers in `data/outputs/metrics.json`.

## Quick start

```bash
make install        # install dependencies
make data           # download the NHTSA corpus via the public API (~few min)
make train          # build splits, train all 3 models, run experiments, write metrics + plots
make app            # serve the app at http://localhost:5000
# or, to verify plumbing in seconds:
make fast
```

`setup.py` writes trained models to `models/`, metrics to `data/outputs/metrics.json`,
and all figures to `data/outputs/plots/`. On the first `make train`, the deep model
downloads pretrained **GloVe** vectors (~822 MB, once) to initialise its embeddings;
they are needed only at training time and are baked into the committed `.pt`, so the
deployed app needs neither GloVe nor an internet connection.

## Deployment

The repo ships a `Dockerfile` (CPU-only torch) that serves inference through gunicorn,
binding to `$PORT` so it runs unchanged on any container host. Models are committed, so
deployment is just "point a host at this repo."

```bash
docker build -t autotriage . && docker run -p 7860:7860 autotriage   # local
```

**Render (free tier) — recommended.** A `render.yaml` blueprint is included. In the
Render dashboard: *New + → Blueprint → connect this repo → Apply*. The free web service
builds the Dockerfile and serves at `https://autotriage-ai.onrender.com` (or similar).

**Hugging Face Space.** The `README.md` front-matter is a Docker-Space config. Note that
HF now requires a **PRO** subscription to host Docker/Gradio Spaces on free CPU; with PRO
it deploys by pushing this repo to a Space.

A scheduled GitHub Action (`.github/workflows/keep-alive.yml`) pings `/health` every 30
min so a free host doesn't sleep during grading — set the `SPACE_URL` repo variable to
your deployed URL.

- `GET /health` → `{"status":"ok","models_loaded":3,...}`
- `POST /api/predict` with `{"text": "...", "model": "deep"}` → prediction + explanation

## Project structure

```
├── README.md                <- this file
├── TECHNICAL_REPORT.md      <- full written report (all rubric sections)
├── PITCH.md                 <- 5-minute pitch script
├── requirements.txt         <- full deps (train + app)
├── requirements-deploy.txt  <- lean inference-only deps (CPU torch)
├── Makefile
├── Dockerfile
├── setup.py                 <- end-to-end pipeline (data → train → eval → experiments)
├── main.py                  <- Flask inference app
├── scripts/
│   ├── make_dataset.py      <- NHTSA API downloader (streams to disk)
│   ├── data.py              <- cleaning, label taxonomy, stratified splits
│   ├── model.py             <- NaiveBaseline / ClassicalModel / TextCNNModel
│   ├── evaluate.py          <- metrics + all plots
│   ├── experiment.py        <- learning curves, noise robustness, gating, head/tail
│   └── eda.py               <- class distribution + length stats
├── models/                  <- trained models (committed) + labels.json
├── data/
│   ├── raw/                 <- API dump (gitignored) + committed sample.csv
│   ├── processed/           <- train/val/test splits (gitignored, regenerated)
│   └── outputs/             <- metrics.json + plots/
├── templates/index.html     <- single-page UI
├── static/                  <- css / js
└── .github/                 <- PR template + keep-alive workflow
```

## Originality & approach

This is new work built for this course. Prior research has classified NHTSA complaints
(mostly for automated defect/recall discovery). What this project contributes:

1. **Interpretable, self-contained triage.** Rather than a black-box classifier, the
   app names the exact words driving each prediction (linear coefficients for the
   classical model; leave-one-token-out saliency for the TextCNN) and attaches a safety
   tier + routing action — the things a triage operator actually needs.
2. **A rigorous three-model comparison** on a compact, de-duplicated 14-class taxonomy,
   with the evaluation focused on the **long tail** of rarer-but-critical systems, where
   accuracy hides failures.
3. **A deployment-oriented experiment suite**: a training-set-size sensitivity study
   (the cold-start question), character-noise robustness (real complaints are messy),
   and confidence-gated abstention for human-in-the-loop routing.

I wrote all code myself; only standard libraries are used (scikit-learn, PyTorch,
Flask). The TextCNN architecture follows Kim (2014). NHTSA data is U.S. public domain.

## Data & citation

- **NHTSA ODI Complaints API** — https://www.nhtsa.gov/nhtsa-datasets-and-apis (public domain)
- Y. Kim, "Convolutional Neural Networks for Sentence Classification," EMNLP 2014.
- J. Pennington, R. Socher, C. Manning, "GloVe: Global Vectors for Word Representation," EMNLP 2014.
