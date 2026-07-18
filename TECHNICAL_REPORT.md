# AutoTriage AI: Classifying Vehicle-Safety Complaints by Affected Component

**Module 2 Project — Natural Language Processing · Technical Report**

---

## Abstract

Every year, U.S. vehicle owners file hundreds of thousands of free-text safety
complaints with the National Highway Traffic Safety Administration (NHTSA). Before
any analysis can happen, each narrative must be routed to the right vehicle system —
brakes, air bags, powertrain, and so on. Today that tagging is largely manual.
This project builds and rigorously evaluates an NLP system that reads a complaint
narrative and predicts the affected component among **14 classes**, trained on
**37,342 real NHTSA complaints** spanning 14 makes. We implement the three required
modeling tiers — a majority-class **naive baseline**, a **classical** TF‑IDF +
Logistic Regression model, and a **TextCNN** deep model with GloVe transfer learning — and compare
them under a metric regime chosen for a long-tailed label distribution (macro‑F1).
Beyond raw accuracy we contribute an *interpretable, deployment-oriented* system: the
app names the words driving each prediction, attaches a safety-tier routing action,
and abstains below a confidence threshold. A focused experiment on training-set-size
sensitivity quantifies the cold-start regime a safety team faces when a newly tracked
model launches. Both real models reach ≈0.77 macro‑F1 (vs a 0.01 naive floor); GloVe
transfer learning is what lifts the neural model from clearly behind the classical
baseline to parity, and closes its cold-start gap entirely.

---

## 1. Problem Statement

NHTSA's Office of Defects Investigation (ODI) is the early-warning system for U.S.
auto safety: owner complaints are the raw signal that, in aggregate, triggers
investigations and recalls. Each complaint is a short, unstructured narrative written
by a lay owner ("the car surged forward on its own in a parking lot"). To be useful,
every narrative must be associated with the vehicle **component/system** it concerns.
That component tag drives everything downstream — trend detection per subsystem,
routing to the right engineering reviewers, and clustering complaints into candidate
defect patterns.

Manual tagging does not scale, is inconsistent between annotators, and is slow exactly
when speed matters (a cluster of brake complaints is a safety emergency). We therefore
frame the task as **single-label multi-class text classification**:

> Given a complaint narrative, predict the primary affected component among 14 classes.

The same problem shape appears anywhere a platform ingests owner feedback about
vehicles — an auto marketplace, a manufacturer's quality inbox, a fleet operator's
maintenance log. A model that reads the narrative and routes it correctly turns a
manual backlog into a real-time triage stream.

**Why this is hard.** Owners are not engineers: they describe symptoms, not systems
("it makes a grinding noise when I stop" → brakes). Several components are semantically
adjacent and easily confused — *engine* vs *powertrain* vs *fuel/propulsion* all
involve "power" and "stalling"; *electrical* is a catch-all that overlaps with almost
everything. And the class distribution is long-tailed, so a model can look accurate
while failing on rarer, safety-critical systems.

---

## 2. Data Sources

**Source.** The public **NHTSA ODI Complaints API**
(`https://api.nhtsa.gov/complaints/complaintsByVehicle`), which returns, per
(make, model, model-year), every filed complaint with a free-text `summary`, a
comma-separated `components` field, and safety flags (`crash`, `fire`, injuries,
deaths). NHTSA data is a U.S. Government work in the public domain.

**Collection.** `scripts/make_dataset.py` crawls a diverse grid of **14 makes** ×
popular models × 4 model-years (2014/2017/2020, plus per-make model resolution via the
products endpoint), streaming results to disk and de-duplicating by complaint ID
(`odiNumber`). This diversity is deliberate: sampling many manufacturers and years
prevents any single recall event from dominating the component distribution and
exposes the model to varied phrasing. The raw crawl collected **77,552 unique
complaints**.

**Label derivation.** NHTSA lists one or more components per complaint (67% list
exactly one). We take the **primary (first-listed) component** as the label and
normalise the raw component vocabulary into a compact, coherent **14-class taxonomy**:
near-duplicate codes are merged (`ENGINE AND ENGINE COOLING` → `ENGINE`;
`FUEL SYSTEM`/`GASOLINE` → `FUEL/PROPULSION SYSTEM`), the modern driver-assistance
codes (`FORWARD COLLISION AVOIDANCE`, `LANE DEPARTURE`, `BACK OVER PREVENTION`) are
grouped as `DRIVER ASSISTANCE (ADAS)`, and non-informative codes
(`UNKNOWN OR OTHER`) are dropped.

**Final dataset.** After cleaning, de-duplication, a minimum-length filter (≥ 8
words), and capping each class at 3,500 examples to bound head-class dominance, we
retain **37,342 labelled complaints across 14 classes**. Narratives average **100
words** (median 83). The distribution is long-tailed: seven head classes sit at the
3,500 cap while the smallest, `DRIVER ASSISTANCE (ADAS)`, has 1,234 — an imbalance
ratio of ≈ 2.8×. We split **70 / 15 / 15** stratified by label into
**26,138 train / 5,602 validation / 5,602 test**. A class-balanced 400-row
`data/raw/sample.csv` is committed for inspection; the full corpus is reproducible
from the API via `make data`.

The 14 classes: ELECTRICAL SYSTEM, ENGINE, POWER TRAIN, STEERING, SERVICE BRAKES,
FUEL/PROPULSION SYSTEM, AIR BAGS, DRIVER ASSISTANCE (ADAS), EXTERIOR LIGHTING,
VISIBILITY/WIPER, STRUCTURE, VEHICLE SPEED CONTROL, SUSPENSION, SEATS/SEAT BELTS.

---

## 3. Related Work

**Complaint and defect mining.** A body of work applies NLP to NHTSA/consumer
complaint text, historically to surface emerging defects and predict recalls. Classic
approaches use bag-of-words / TF‑IDF features with linear classifiers or topic models
(LDA) to cluster complaints and flag anomalies; more recent work fine-tunes
transformer encoders for defect classification. These efforts largely target
*recall/defect discovery* (an unsupervised or weakly-supervised trend-detection goal)
rather than supervised routing of a narrative to a component taxonomy.

**Text classification methods.** Our modeling tiers follow the standard progression in
text classification. TF‑IDF with n-grams plus a linear model (logistic regression /
linear SVM) remains a famously strong, interpretable baseline (Wang & Manning, 2012).
For the neural tier we use the **TextCNN** of Kim (2014), which applies parallel
convolutional filters as learned n-gram detectors — a compact architecture that is
competitive on short-to-medium documents and cheap enough to deploy on CPU. Large
pretrained transformers (e.g. BERT, DistilBERT) typically define the accuracy ceiling
but at a much larger parameter and latency cost.

**What is new here.** Relative to prior complaint-mining work, this project (1) frames
the task as a clean supervised 14-class routing problem on a compact, de-duplicated
taxonomy; (2) centres the evaluation on the **long tail** of rarer-but-critical
components rather than headline accuracy; and (3) delivers an *interpretable,
deployment-ready* triage tool — word-level explanations, safety-tier routing, and
confidence-gated abstention — instead of a black-box label. Our aim is greater insight
and usability rather than chasing the transformer SOTA ceiling; we discuss that
trade-off explicitly in §9.

---

## 4. Evaluation Strategy & Metrics

**The metric choice is the most important evaluation decision in this project**, and it
is driven by the long-tailed label distribution.

- **Why not accuracy.** With imbalanced classes, plain accuracy rewards a model for
  getting the frequent classes right while ignoring rare ones. The naive majority
  baseline makes this concrete: it can post a non-trivial accuracy while scoring **0
  recall** on 13 of 14 components. Accuracy alone would call that "not terrible"; for
  a safety-routing system it is useless.
- **Headline metric: macro‑F1.** Macro‑F1 averages the per-class F1 with **equal
  weight per class**, so failing on `AIR BAGS` (rare) is penalised as heavily as
  failing on `ELECTRICAL SYSTEM` (common). This directly matches the business need:
  every safety system matters, not just the popular ones.
- **Supporting metrics.** We also report **accuracy** (overall correctness),
  **weighted‑F1** (F1 weighted by support, a middle ground), and **full per-class
  precision/recall/F1** with confusion matrices, because *which* components get
  confused is itself a finding (see Error Analysis).
- **Deployment metrics.** Because a triage system can defer uncertain cases to a
  human, we additionally measure **accuracy vs. coverage** under confidence-gated
  abstention (§7.3).

All models are selected/tuned on the validation split and reported **once** on the
untouched test split. The deep model's early stopping is driven by **validation
macro‑F1**, consistent with the headline metric.

---

## 5. Modeling Approach

### 5.1 Data processing pipeline (with rationale)

Every step in `scripts/data.py` earns its place:

1. **Primary-label derivation** — take the first-listed component; rationale: 67% of
   complaints are single-component, and the first-listed code is NHTSA's primary
   association. Keeps the task a clean single-label problem.
2. **Taxonomy normalisation** — merge near-duplicate codes and group ADAS; rationale:
   raw codes have redundant/rare variants that fragment supervision and confuse
   evaluation. A compact 14-class taxonomy gives coherent, learnable classes.
3. **Text cleaning** — lowercase; strip NHTSA editorial tokens (`TL*`, `*TR`), PII
   redactions (`XXX`), VINs, phone numbers, URLs/emails; keep alphanumerics, slashes,
   hyphens. Rationale: those artefacts are noise that a bag-of-words model would treat
   as spurious features and that add nothing for the CNN.
4. **Minimum-length filter (≥ 8 words)** — drop empty/degenerate narratives that carry
   no signal.
5. **De-duplication** on cleaned text — identical narratives (re-files, boilerplate)
   would leak between splits and inflate scores.
6. **Class cap (3,500)** — bound head-class dominance so the model and the macro metric
   are not swamped by the two or three largest components.
7. **Stratified split (70/15/15)** — preserve the class distribution in every split so
   validation and test are representative, especially for tail classes.

The **classical** and **deep** models share this cleaned text but diverge in
featurisation: TF‑IDF n-gram vectors for the linear model; a learned integer
vocabulary + embeddings for the CNN.

### 5.2 Models evaluated (and why each)

**(a) Naive baseline — majority class** (`NaiveBaseline`). Predicts the single most
frequent class for every input and exposes the training class priors as its
probabilities. *Rationale:* establishes the accuracy floor and demonstrates precisely
why accuracy is misleading here — it is the reference every real model must beat on
macro‑F1.

**(b) Classical ML — TF‑IDF + Logistic Regression** (`ClassicalModel`). Word 1–2 gram
TF‑IDF (sublinear tf, `min_df=3`, up to 40k features) into a multinomial logistic
regression with **balanced class weights**. *Rationale:* a strong, fast, and — crucially
— **interpretable** baseline. The linear coefficients name the exact tokens that push a
complaint toward each component, which powers the app's explanations and lets us audit
what the model learned. Class weighting counteracts the residual imbalance.

**(c) Deep learning — TextCNN with GloVe transfer learning** (`TextCNNModel`,
*deployed*). An embedding layer **initialised with pretrained 200‑d GloVe vectors**
(Pennington et al., 2014; ~85% of our vocabulary covered) and fine-tuned during
training feeds four parallel 1‑D convolutions of widths 2/3/4/5 (160 filters each)
acting as learned n‑gram detectors; max-over-time pooling keeps the strongest
activation per filter; dropout (0.4) then a linear classifier. Trained with Adam and
**class-weighted cross-entropy**, early-stopped on validation macro‑F1.
*Rationale:* the CNN learns phrase-level, order-sensitive cues ("lost power", "goes to
the floor") that bag-of-words misses, and the GloVe initialisation is a lightweight
form of **transfer learning** — it injects general-language knowledge the model cannot
learn from ~26k complaints alone, which is what lifts it from clearly behind the
classical baseline to parity with it (§6). The pretrained vectors are needed only at
training time; the learned embedding matrix is baked into the ~15 MB `.pt`, so the
deployed model stays tiny, CPU-fast, and fully self-contained. We use GloVe rather than
fine-tuning a large transformer to keep the system lightweight and deployable; §9
discusses that trade-off.

### 5.3 Hyperparameter tuning strategy

We tuned on the validation split, not the test split. For the **classical** model the
key knobs are the TF‑IDF vocabulary (`ngram_range`, `min_df`, `max_features`) and the
LogReg regularisation `C`; we favour 1–2 grams (phrase cues like "check engine") and a
moderately low `C` for generalisation, with `class_weight="balanced"` fixed by the
imbalance. For the **TextCNN**, filter widths {2,3,4,5} and count (160) follow Kim
(2014)'s well-established design; the biggest lever we found was the **embedding
initialisation** — switching from random to pretrained GloVe (200‑d) added roughly two
macro‑F1 points, and helped most in the low-data regime (§7). We also probed embedding
dimension and vocabulary cut-off: an enhanced-capacity variant (mean+max pooling plus a
hidden layer) and a larger `min_freq=1` vocabulary each *slightly hurt* on validation,
so we kept the simpler, better-generalising design. Dropout (0.4), Adam (1e‑3), and
**early stopping on validation macro‑F1** (patience 5) are the remaining controls,
which prevent overfitting without a manual epoch search.

---

## 6. Results

### 6.1 Model comparison (held-out test set, n = 5,602)

| Model | Accuracy | **Macro‑F1** | Weighted‑F1 |
|---|---|---|---|
| Naive (majority class) | 0.094 | **0.012** | 0.016 |
| Classical (TF‑IDF + LogReg) | 0.774 | **0.768** | 0.775 |
| Deep (TextCNN + GloVe) *(deployed)* | 0.771 | **0.762** | 0.770 |

Two things stand out. First, **the naive baseline validates our metric choice**: it
posts 0.094 accuracy but 0.012 macro‑F1, because it gets one class fully right and the
other 13 entirely wrong. Any model must be judged on macro‑F1, and both real models
clear the floor by ~0.75. Second, **the classical and deep models are a statistical
tie** (0.768 vs 0.762 macro‑F1; 0.774 vs 0.771 accuracy — a difference within run-to-run
noise). This is itself a finding: with GloVe transfer learning the neural model reaches
parity with a very strong linear baseline, but does not surpass it, because the
component signal is largely lexical ("brake", "airbag", "steering") — exactly what
TF‑IDF captures best. See `data/outputs/plots/model_comparison.png`.

### 6.2 Per-class performance

Both models are **well-balanced across the long tail** — the head/tail F1 gap is only
~0.02–0.03 (§7.4), a payoff from capping head classes and class-weighting. Per-class
F1 (deployed deep model) ranges from strong to hard:

- **Easiest:** AIR BAGS 0.91, SEATS/SEAT BELTS 0.88, EXTERIOR LIGHTING 0.87,
  VISIBILITY/WIPER 0.86 — these have distinctive vocabulary ("airbag", "seat belt").
- **Hardest:** VEHICLE SPEED CONTROL 0.53, ELECTRICAL SYSTEM 0.57, DRIVER ASSISTANCE
  (ADAS) 0.61 — semantically diffuse or overlapping systems.

### 6.3 What gets confused (and why)

The dominant confusions (from `confusion_deep.png`) are all **semantically adjacent
systems**: ENGINE ↔ POWER TRAIN (87 test cases both directions), POWER TRAIN ↔
ELECTRICAL SYSTEM (72), STRUCTURE ↔ VISIBILITY/WIPER (48), SERVICE BRAKES ↔ ADAS (33),
STEERING ↔ SUSPENSION (28), ENGINE ↔ FUEL/PROPULSION (28). "Electrical system" is the
worst offender because it is a genuine catch-all that co-occurs with almost everything
("check engine light", "sensor fault"). These are not random errors — they mirror real
ambiguity in how a symptom maps to a subsystem, which motivates the error analysis and
the multi-label direction in future work.

## 7. Experiment Write-Up

### 7.1 Primary experiment — training-set-size sensitivity (the cold-start question)

**Plan.** A real deployment starts cold: when a platform begins tracking a new vehicle
or launches the model, it has few labelled complaints. How much data does each model
need? We retrain every model on stratified subsets (5% → 100% of the training set) and
score macro‑F1 on the **fixed** test set. Motivated, controlled, and directly relevant
to deployment economics.

**Results (macro‑F1).**

| Train size | 1,306 | 2,613 | 6,534 | 13,069 | 26,138 |
|---|---|---|---|---|---|
| Naive | 0.012 | 0.012 | 0.012 | 0.012 | 0.012 |
| Classical | 0.707 | 0.726 | 0.748 | 0.762 | **0.768** |
| Deep (TextCNN + GloVe) | 0.706 | 0.732 | 0.753 | 0.762 | 0.765 |

**Interpretation.** This is the experiment where GloVe earns its keep. In an earlier
from-scratch version of the CNN (no pretrained embeddings), the deep model trailed the
classical model by **11 macro‑F1 points at the smallest size** (0.596 vs 0.707) — it
was starved for data. **With GloVe transfer learning that gap disappears**: at 1,306
examples the two are tied (0.706 vs 0.707), and in the mid-data regime the deep model
actually *edges ahead* (0.732 vs 0.726 at 2.6k; 0.753 vs 0.748 at 6.5k). Both curves
are still rising gently at full data, with the classical model finishing marginally on
top. See `data/outputs/plots/learning_curve.png`.

**Recommendation.** Pretrained embeddings should be considered mandatory for the neural
model in any low-resource / cold-start deployment; without them the classical model is
strictly preferable. At full data either model is a fine choice.

### 7.2 Robustness to noisy text

Real complaints are full of typos and inconsistent casing, so we inject character-level
noise (deletion/substitution/insertion) at increasing rates and re-score.

| Char-noise rate | 0% | 5% | 10% | 20% | 30% |
|---|---|---|---|---|---|
| Classical | 0.768 | 0.724 | 0.684 | 0.555 | 0.418 |
| Deep (TextCNN + GloVe) | 0.762 | 0.704 | 0.617 | 0.431 | 0.276 |

**Interpretation.** The classical model is **notably more robust** — at 10% noise it
holds 0.684 vs the CNN's 0.617, and the gap widens with noise. Reason: the word-level
CNN vocabulary sends every typo to `<unk>`, discarding signal, whereas TF‑IDF's larger
n‑gram vocabulary degrades more gracefully. This is the clearest argument *for* the
classical model and *for* the future-work move to subword embeddings.

### 7.3 Confidence-gated abstention (deployment triage)

Because a triage system can defer, we sweep a confidence threshold on the deployed
model: answer only above `t`, route the rest to a human.

| Threshold `t` | 0.0 | 0.5 | 0.7 | 0.9 |
|---|---|---|---|---|
| Coverage (fraction answered) | 1.00 | 0.89 | 0.77 | 0.60 |
| Accuracy on answered | 0.771 | 0.822 | 0.864 | 0.912 |

**Interpretation & recommendation.** Abstention buys large accuracy gains: answering the
most-confident **77%** of complaints reaches **86% accuracy**, and the top **60%** reach
**91%**. A production system should auto-route above a tuned threshold and escalate the
uncertain remainder — turning a 0.77-accuracy model into a high-precision autorouter
plus a managed human queue. See `data/outputs/plots/confidence_coverage.png`.

### 7.4 Head vs. tail

Splitting classes into the 5 most frequent (head) and 9 rarer (tail): classical scores
head 0.782 / tail 0.761; deep scores head 0.781 / tail 0.751. The **~0.02–0.03 gap is
small**, confirming that the head-class cap plus class-weighted training kept the
rare-but-critical systems from being neglected.

## 8. Error Analysis

We surfaced the deployed model's **most confident mistakes** (all at ≈1.00 confidence)
— the dangerous kind, where the model is both wrong and sure. Five representative cases:

1. **"noise on front passenger side suspension"** — *true: STEERING, pred: SUSPENSION.*
   **Root cause:** label ambiguity between two adjacent chassis systems; the narrative
   literally says "suspension" while NHTSA's primary code is steering. **Mitigation:**
   move to multi-label prediction, or merge the two into a "steering/suspension" super-
   class; abstain on very short narratives (this one is 6 words).

2. **"Takata inflators … not one airbag deployed [after crash]"** — *true: STEERING,
   pred: AIR BAGS.* **Root cause:** a multi-component complaint whose salient text is
   entirely about airbags, but whose primary NHTSA code is steering. The model reads the
   text correctly; the single-label framing is what's wrong. **Mitigation:** multi-label
   output so both AIR BAGS and STEERING can be returned.

3. **"check engine light … smelling gasoline in the cabin"** — *true: ELECTRICAL SYSTEM,
   pred: FUEL/PROPULSION.* **Root cause:** a strong lexical cue ("gasoline") hijacks the
   prediction over the true electrical fault. **Mitigation:** richer context modeling
   (the fix was electrical), or multi-label; down-weight single-token triggers.

4. **"t-boned … airbags did not deploy … door smashed in"** — *true: STRUCTURE, pred:
   AIR BAGS.* **Root cause:** crash narratives mention many systems; the vivid "airbags
   did not deploy" dominates the true structural-damage label. **Mitigation:** add crash-
   context features / multi-label; these high-severity cases warrant human review anyway
   (route via the safety-critical tier).

5. **"front lights are extremely bright for other drivers … glare"** — *true:
   VISIBILITY/WIPER, pred: EXTERIOR LIGHTING.* **Root cause:** taxonomy overlap — glare
   is a *visibility* problem described entirely with *lighting* vocabulary.
   **Mitigation:** clearer annotation guidance / a hierarchical taxonomy that lets
   "headlight glare" live under both.

**Cross-cutting theme:** the confident errors are overwhelmingly (a) **multi-component
complaints** where a single primary label can't capture the text, and (b) **genuinely
adjacent/ambiguous systems**. Both point to the same highest-value fix — **multi-label
classification with calibrated per-label thresholds** — rather than more model capacity.

---

## 9. Commercial Viability

**Is this suitable for real-world use? Partly yes, with the right framing.** The system
is a genuine fit for a *human-in-the-loop triage* product, not an autonomous
decision-maker. Concretely:

- **Where it fits.** Any organisation ingesting free-text vehicle feedback — a
  manufacturer's quality/warranty inbox, an auto marketplace like the sponsor's
  platform, a fleet operator's maintenance logs, or an insurer's claims text — faces
  the same manual-tagging bottleneck. A ~0.77 macro-F1 router that also abstains on
  low-confidence cases can auto-tag the confident majority and escalate the rest,
  cutting triage cost materially while keeping a human on the hard calls.
- **Why the design is deployable.** The model is a ~15 MB artefact running CPU
  inference in milliseconds with no GPU — cheap to host and scale. It is interpretable
  (word-level evidence + a safety tier), which is essential for adoption by safety
  reviewers who must justify decisions.
- **Why not fully autonomous.** Safety is high-stakes and the tail classes are the ones
  that matter most; at ~0.77 macro-F1 the model is a productivity multiplier, not a
  replacement for expert judgement. The confidence-gating result (§7) is the mechanism
  that makes this safe: answer automatically only where the model is reliable.
- **The model-choice trade-off.** Our evaluation shows a strong classical baseline is
  on par with the neural model, so a lean production system could ship the classical
  model alone. We deploy the GloVe TextCNN to demonstrate the deep approach and because
  it generalises semantically, but a cost-sensitive buyer could run the classical model
  at a fraction of the footprint — an honest finding that de-risks the product.

**Verdict:** commercially viable as a triage-assist feature; not as an unsupervised
safety authority.

## 10. Ethics Statement

- **Data & privacy.** NHTSA complaints are public-domain records, but narratives can
  contain personal details. Our pipeline strips VINs, phone numbers, emails, and
  redaction artefacts before modelling, and we commit only a small sample plus a
  reproducible download script rather than re-publishing the corpus.
- **Safety-critical misuse.** A wrong or over-trusted prediction could misroute a
  genuine safety defect. The system is explicitly positioned as a triage aid with
  human review and confidence-gated abstention; it must not be used to *dismiss*
  complaints or to make recall decisions autonomously.
- **Bias & representativeness.** The corpus skews toward high-volume U.S. makes/models
  and English-language, self-selected complainants; performance may be worse for
  under-represented vehicles, populations, or non-English text. Per-class (head/tail)
  reporting is our first guard against silently failing rare-but-critical systems.
- **Transparency.** Every prediction ships with the evidence behind it and a calibrated
  confidence, so reviewers can audit and overrule the model rather than defer to it.

## 11. Conclusions & Future Work

**Conclusions.** On 37k real NHTSA complaints, a disciplined three-model comparison for
14-way component routing shows: (1) the task is very learnable from narrative text —
both the classical TF‑IDF model and the GloVe-initialised TextCNN reach ~0.77 macro-F1,
crushing the 0.01 naive floor; (2) **the metric choice matters more than the model** —
accuracy alone would have hidden the naive baseline's uselessness and the tail-class
behaviour; (3) **transfer learning (GloVe) is what makes the neural model competitive**,
closing a two-point gap and helping most when labels are scarce; and (4) a simple,
interpretable classical model is a remarkably strong, cheap baseline that a real product
could ship. The deployed app turns these models into a usable triage tool with
word-level explanations, safety-tier routing, and confidence-gated abstention.

**Future work (with another semester).**

- **Multi-label routing.** ~33% of complaints touch several systems; move from
  primary-component single-label to true multi-label prediction with per-label
  thresholds.
- **Subword / transformer encoders.** Fine-tune a domain-adapted DistilBERT or use
  fastText subword embeddings to (a) chase the accuracy ceiling and (b) fix the
  character-noise brittleness the robustness study exposed.
- **Calibration & abstention policy.** Temperature-scale the probabilities and learn a
  cost-sensitive abstention threshold per class (missing an air-bag defect ≠ missing a
  wiper complaint).
- **Temporal defect detection.** Layer per-component time-series anomaly detection on
  top of the router to surface *emerging* defect clusters — the actual NHTSA mission.
- **Active learning for the tail.** Prioritise human labelling of low-confidence
  tail-class complaints to lift the components that matter most per unit of annotation.

## References

1. Y. Kim. "Convolutional Neural Networks for Sentence Classification." *EMNLP*, 2014.
2. J. Pennington, R. Socher, C. D. Manning. "GloVe: Global Vectors for Word
   Representation." *EMNLP*, 2014.
3. S. Wang, C. D. Manning. "Baselines and Bigrams: Simple, Good Sentiment and Topic
   Classification." *ACL*, 2012.
4. NHTSA Office of Defects Investigation. Complaints API / datasets.
   https://www.nhtsa.gov/nhtsa-datasets-and-apis (public domain).

---

*Reproduce all numbers and figures in this report with `make train` (writes
`data/outputs/metrics.json` and `data/outputs/plots/`).*
