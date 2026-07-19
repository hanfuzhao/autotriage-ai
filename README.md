---
title: AutoTriage AI
emoji: 🚗
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# AutoTriage AI: Is Your Car Problem a Safety Issue?

Module 2 Project, Natural Language Processing. Built on real NHTSA owner complaints.

Most drivers can't tell a real safety defect from a harmless quirk. You hear a noise, a
warning light comes on, the car does something that doesn't feel right, and you are left
guessing whether it is dangerous or nothing. AutoTriage AI reads your plain-language
description and tells you which of 14 car systems it points to, how serious that system
tends to be, and whether it is worth reporting to NHTSA. It learned from the same owner
complaints that, added up, trigger recalls, so a report you file could help catch the next
defect early.

- Live demo: https://autotriage-ai-unfvnsiy6a-uc.a.run.app (Google Cloud Run)
- Task: single-label text classification over 14 vehicle-component classes
- Data: the public NHTSA ODI Complaints API, real owner narratives in the public domain

## What it does

You describe what your car did. The app returns the most likely system, a confidence
score, the other systems it could be, the exact words that drove the call so you can
sanity-check it, a safety tier with a plain next step, and a side-by-side view of how all
three models read the same text.

## The three required models

| Approach | Where | Trained file | What it is |
|---|---|---|---|
| Naive baseline | `scripts/model.py::NaiveBaseline` | `models/naive.pkl` | Majority-class predictor, the accuracy floor |
| Classical ML | `scripts/model.py::ClassicalModel` | `models/classical.pkl` | TF-IDF, 1 to 2 grams, with class-weighted Logistic Regression |
| Deep learning, deployed | `scripts/model.py::TextCNNModel` | `models/deep_textcnn.pt` | TextCNN (Kim 2014) with GloVe transfer learning (Pennington 2014) |

All three are implemented, trained, and evaluated. The TextCNN is the deployed model.
It is only about 15 MB, so it is committed with the repo and the app runs on its own with
no external model hosting.

## Headline results (held-out test set)

| Model | Accuracy | Macro-F1 | Weighted-F1 |
|---|---|---|---|
| Naive (majority class) | 0.094 | 0.012 | 0.016 |
| Classical (TF-IDF + LogReg) | 0.774 | 0.768 | 0.775 |
| Deep (TextCNN + GloVe), deployed | 0.771 | 0.762 | 0.770 |

The component distribution is long-tailed, so macro-F1 is the metric that matters, not
accuracy. Every safety system counts equally. The naive baseline makes the point: it
scores 0.094 accuracy but only 0.012 macro-F1. GloVe transfer learning is what pulls the
neural model up to the classical baseline and removes its cold-start gap. See
`TECHNICAL_REPORT.md` section 7. Full numbers live in `data/outputs/metrics.json`.

## Quick start

```bash
make install        # install dependencies
make data           # download the NHTSA corpus from the public API
make train          # build splits, train all 3 models, run experiments, write metrics and plots
make app            # serve the app at http://localhost:5000
make fast           # tiny run in seconds to check the plumbing
```

`setup.py` writes trained models to `models/`, metrics to `data/outputs/metrics.json`,
and figures to `data/outputs/plots/`. The first `make train` downloads pretrained GloVe
vectors once to initialise the deep model's embeddings. Those vectors are only used at
training time and are baked into the committed `.pt`, so the deployed app needs neither
GloVe nor a network connection.

## Deployment

The repo ships a `Dockerfile` with CPU-only torch that serves inference through gunicorn
and binds to `$PORT`, so it runs unchanged on any container host. The models are
committed, so deploying is just pointing a host at this repo.

```bash
docker build -t autotriage . && docker run -p 7860:7860 autotriage   # local
```

The live demo runs on Google Cloud Run. There is also a `render.yaml` blueprint if you
prefer Render's free tier: in the Render dashboard pick New, Blueprint, connect this repo,
Apply. The HF Space front-matter at the top of this file still works too, though HF now
needs a PRO plan to host Docker Spaces on free CPU.

A scheduled GitHub Action in `.github/workflows/keep-alive.yml` pings `/health` every 30
minutes so a free host does not sleep during grading. Set the `SPACE_URL` repo variable to
the deployed URL.

- `GET /health` returns `{"status":"ok","models_loaded":3,...}`
- `POST /api/predict` with `{"text": "...", "model": "deep"}` returns the prediction and explanation

## Project structure

```
README.md                 this file
TECHNICAL_REPORT.md       full written report
PITCH.md                  5-minute pitch script
PITCH.pptx                pitch slide deck
requirements.txt          full deps for training and app
requirements-deploy.txt   lean inference-only deps, CPU torch
Makefile
Dockerfile
setup.py                  end-to-end pipeline: data, train, eval, experiments
main.py                   Flask inference app
scripts/
  make_dataset.py         NHTSA API downloader, streams to disk
  data.py                 cleaning, label taxonomy, stratified splits
  model.py                NaiveBaseline, ClassicalModel, TextCNNModel
  evaluate.py             metrics and plots
  experiment.py           learning curves, noise robustness, gating, head/tail
  eda.py                  class distribution and length stats
  make_slides.py          builds the pitch deck
models/                   trained models and labels.json
data/
  raw/                    API dump (gitignored) plus a committed sample.csv
  processed/              train/val/test splits, regenerated
  outputs/                metrics.json and plots
templates/index.html      single-page UI
static/                   css and js
.github/                  PR template and keep-alive workflow
```

## Originality and approach

This is new work built for this course. Earlier research has classified NHTSA complaints,
mostly to help regulators find defects or predict recalls. This project points the same
capability at the person who actually notices the problem first, the owner. What it adds:

1. An owner-facing, interpretable answer. Instead of a black box, the app names the exact
   words behind each prediction, using the linear coefficients for the classical model and
   leave-one-token-out saliency for the TextCNN, and it attaches a safety tier and a plain
   next step. An owner needs to trust the call, and being able to see why is what makes it
   trustworthy.
2. A careful three-model comparison on a compact 14-class taxonomy, with the evaluation
   aimed at the long tail of rarer but critical systems like airbags and brakes, where
   accuracy hides failures and where a false all-clear would be the most harmful.
3. A deployment-minded set of experiments: a training-set-size study for the cold-start
   case, character-noise robustness because real complaints are messy, and
   confidence-gated abstention so the tool can say "not sure" instead of falsely
   reassuring someone.

I wrote the code myself and used only standard libraries: scikit-learn, PyTorch, Flask.
The TextCNN follows Kim 2014. NHTSA data is U.S. public domain.

## Acknowledgments

I built this project for the course. I used an AI coding assistant (Anthropic's Claude)
to help draft and refactor some of the code and to help edit the writing. I reviewed,
tested, and edited everything myself before submitting.

## Data and citations

- NHTSA ODI Complaints API, https://www.nhtsa.gov/nhtsa-datasets-and-apis, public domain
- Y. Kim, "Convolutional Neural Networks for Sentence Classification," EMNLP 2014
- J. Pennington, R. Socher, C. Manning, "GloVe: Global Vectors for Word Representation," EMNLP 2014
