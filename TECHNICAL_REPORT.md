# AutoTriage AI: Classifying Vehicle-Safety Complaints by Affected Component

Module 2 Project. Natural Language Processing. Technical Report.

---

## Abstract

Every year, U.S. vehicle owners file hundreds of thousands of free-text safety
complaints with the National Highway Traffic Safety Administration (NHTSA). Before
any analysis can happen, each narrative has to be routed to the right vehicle system:
brakes, air bags, powertrain, and so on. Right now that tagging is mostly done by hand.
This project builds and evaluates an NLP system that reads a complaint
narrative and predicts the affected component among **14 classes**. It is trained on
37,342 real NHTSA complaints spanning 14 makes. We implement the three required
modeling tiers, a majority-class naive baseline, a classical TF-IDF +
Logistic Regression model, and a TextCNN deep model with GloVe transfer learning, then compare
them under a metric regime chosen for a long-tailed label distribution, macro-F1.
Beyond raw accuracy we contribute an interpretable, deployment-oriented system. The
app names the words driving each prediction, attaches a safety-tier routing action,
and abstains below a confidence threshold. A focused experiment on training-set-size
sensitivity quantifies the cold-start regime a safety team faces when a newly tracked
model launches. Both real models reach about 0.77 macro-F1 against a 0.01 naive floor. GloVe
transfer learning is what lifts the neural model from clearly behind the classical
baseline up to parity, and it closes the cold-start gap entirely.

---

## 1. Problem Statement

NHTSA's Office of Defects Investigation (ODI) is the early-warning system for U.S.
auto safety. Owner complaints are the raw signal that, in aggregate, triggers
investigations and recalls. Each complaint is a short, unstructured narrative written
by a lay owner, for example "the car surged forward on its own in a parking lot". To be useful,
every narrative has to be tied to the vehicle **component/system** it concerns.
That component tag drives everything downstream: trend detection per subsystem,
routing to the right engineering reviewers, and clustering complaints into candidate
defect patterns.

Manual tagging does not scale, it is inconsistent between annotators, and it is slow exactly
when speed matters. A cluster of brake complaints is a safety emergency. So we
frame the task as **single-label multi-class text classification**:

> Given a complaint narrative, predict the primary affected component among 14 classes.

The same problem shows up anywhere a platform ingests owner feedback about
vehicles, whether that is an auto marketplace, a manufacturer's quality inbox, or a fleet operator's
maintenance log. A model that reads the narrative and routes it correctly turns a
manual backlog into a real-time triage stream.

**Why this is hard.** Owners are not engineers. They describe symptoms, not systems.
"It makes a grinding noise when I stop" means brakes. Several components are semantically
adjacent and easily confused. Engine, powertrain, and fuel/propulsion all
involve "power" and "stalling", and electrical is a catch-all that overlaps with almost
everything. On top of that the class distribution is long-tailed, so a model can look accurate
while failing on rarer, safety-critical systems.

---

## 2. Data Sources

**Source.** We use the public **NHTSA ODI Complaints API**
(`https://api.nhtsa.gov/complaints/complaintsByVehicle`). For each
make, model, and model-year it returns every filed complaint with a free-text `summary`, a
comma-separated `components` field, and safety flags for crash, fire, injuries, and
deaths. NHTSA data is a U.S. Government work in the public domain.

**Collection.** `scripts/make_dataset.py` crawls 14 makes and their popular models
across three model-years, 2014, 2017, and 2020, resolving the exact model names per make
through the products endpoint. It streams results to disk and de-duplicates by complaint
ID `odiNumber`. The diversity is deliberate. Sampling many manufacturers and years
keeps any single recall event from dominating the component distribution, and it
exposes the model to varied phrasing. The raw crawl collected **77,552 unique
complaints**.

**Label derivation.** NHTSA lists one or more components per complaint, and 67% list
exactly one. We take the primary, first-listed component as the label and
normalise the raw component vocabulary into a compact 14-class taxonomy.
Near-duplicate codes are merged, so `ENGINE AND ENGINE COOLING` becomes `ENGINE` and
`FUEL SYSTEM`/`GASOLINE` becomes `FUEL/PROPULSION SYSTEM`. The modern driver-assistance
codes `FORWARD COLLISION AVOIDANCE`, `LANE DEPARTURE`, and `BACK OVER PREVENTION` are
grouped as `DRIVER ASSISTANCE (ADAS)`, and non-informative codes like
`UNKNOWN OR OTHER` are dropped.

**Final dataset.** After cleaning, de-duplication, a minimum-length filter of at least 8
words, and capping each class at 3,500 examples to bound head-class dominance, we
retain **37,342 labelled complaints across 14 classes**. Narratives average 100
words, with a median of 83. The distribution is long-tailed. Seven head classes sit at the
3,500 cap while the smallest, `DRIVER ASSISTANCE (ADAS)`, has 1,234. That is an imbalance
ratio of about 2.8x. We split **70 / 15 / 15** stratified by label into
**26,138 train / 5,602 validation / 5,602 test**. A class-balanced 400-row
`data/raw/sample.csv` is committed for inspection, and the full corpus is reproducible
from the API via `make data`.

The 14 classes: ELECTRICAL SYSTEM, ENGINE, POWER TRAIN, STEERING, SERVICE BRAKES,
FUEL/PROPULSION SYSTEM, AIR BAGS, DRIVER ASSISTANCE (ADAS), EXTERIOR LIGHTING,
VISIBILITY/WIPER, STRUCTURE, VEHICLE SPEED CONTROL, SUSPENSION, SEATS/SEAT BELTS.

![Component class distribution. The data is long-tailed, which is why we report macro-F1.](data/outputs/plots/class_distribution.png)

---

## 3. Related Work

**Complaint and defect mining.** A body of work applies NLP to NHTSA and consumer
complaint text, historically to surface emerging defects and predict recalls. Classic
approaches use bag-of-words or TF-IDF features with linear classifiers, or topic models
such as LDA, to cluster complaints and flag anomalies. More recent work fine-tunes
transformer encoders for defect classification. Most of these efforts target
recall and defect discovery, an unsupervised or weakly-supervised trend-detection goal,
rather than supervised routing of a narrative to a component taxonomy.

**Text classification methods.** Our modeling tiers follow the standard progression in
text classification. TF-IDF with n-grams plus a linear model, either logistic regression or
a linear SVM, remains a famously strong and interpretable baseline (Wang & Manning, 2012).
For the neural tier we use the **TextCNN** of Kim (2014), which applies parallel
convolutional filters as learned n-gram detectors. It is a compact architecture that is
competitive on short-to-medium documents and cheap enough to deploy on CPU. Large
pretrained transformers such as BERT and DistilBERT typically define the accuracy ceiling,
but they cost far more in parameters and latency.

**What is new here.** Relative to prior complaint-mining work, this project does three
things. First, it frames
the task as a clean supervised 14-class routing problem on a compact, de-duplicated
taxonomy. Second, it centres the evaluation on the long tail of rarer-but-critical
components rather than headline accuracy. Third, it delivers an interpretable,
deployment-ready triage tool with word-level explanations, safety-tier routing, and
confidence-gated abstention, instead of a black-box label. Our aim is greater insight
and usability rather than chasing the transformer SOTA ceiling, and we discuss that
trade-off explicitly in section 9.

---

## 4. Evaluation Strategy & Metrics

The metric choice is the most important evaluation decision in this project, and it
follows directly from the long-tailed label distribution.

- **Why not accuracy.** With imbalanced classes, plain accuracy rewards a model for
  getting the frequent classes right while ignoring rare ones. The naive majority
  baseline makes this concrete. It can post a non-trivial accuracy while scoring 0
  recall on 13 of 14 components. Accuracy alone would call that "not terrible". For
  a safety-routing system it is useless.
- **Headline metric: macro-F1.** Macro-F1 averages the per-class F1 with equal
  weight per class, so failing on the rare `AIR BAGS` class is penalised as heavily as
  failing on the common `ELECTRICAL SYSTEM` class. This matches the business need:
  every safety system matters, not just the popular ones.
- **Supporting metrics.** We also report accuracy for overall correctness,
  weighted-F1 as a middle ground that weights F1 by support, and full per-class
  precision/recall/F1 with confusion matrices, since which components get
  confused turns out to matter (see Error Analysis).
- **Deployment metrics.** Because a triage system can defer uncertain cases to a
  human, we also measure accuracy against coverage under confidence-gated
  abstention (section 7.3).

All models are selected and tuned on the validation split, then reported once on the
untouched test split. The deep model's early stopping is driven by validation
macro-F1, consistent with the headline metric.

---

## 5. Modeling Approach

### 5.1 Data processing pipeline (with rationale)

Every step in `scripts/data.py` is there for a reason:

1. **Primary-label derivation.** Take the first-listed component. 67% of
   complaints are single-component, and the first-listed code is NHTSA's primary
   association, so this keeps the task a clean single-label problem.
2. **Taxonomy normalisation.** Merge near-duplicate codes and group ADAS. Raw
   codes have redundant and rare variants that fragment supervision and confuse
   evaluation, whereas a compact 14-class taxonomy gives coherent, learnable classes.
3. **Text cleaning.** Lowercase, strip NHTSA editorial tokens like `TL*` and `*TR`, PII
   redactions like `XXX`, VINs, phone numbers, URLs, and emails, and keep alphanumerics, slashes,
   and hyphens. Those artefacts are noise. A bag-of-words model would treat
   them as spurious features, and they add nothing for the CNN.
4. **Minimum-length filter of at least 8 words.** Drop empty or degenerate narratives that carry
   no signal.
5. **De-duplication** on cleaned text. Identical narratives, such as re-files and boilerplate,
   would leak between splits and inflate scores.
6. **Class cap of 3,500.** Bound head-class dominance so the model and the macro metric
   are not swamped by the two or three largest components.
7. **Stratified split (70/15/15).** Preserve the class distribution in every split so
   validation and test are representative, especially for tail classes.

The classical and deep models share this cleaned text but diverge in
featurisation. The linear model uses TF-IDF n-gram vectors. The CNN uses a learned integer
vocabulary and embeddings.

### 5.2 Models evaluated (and why each)

**(a) Naive baseline, majority class** (`NaiveBaseline`). Predicts the single most
frequent class for every input and exposes the training class priors as its
probabilities. This establishes the accuracy floor and shows exactly
why accuracy is misleading here. It is the reference every real model has to beat on
macro-F1.

**(b) Classical ML, TF-IDF + Logistic Regression** (`ClassicalModel`). Word 1-2 gram
TF-IDF with sublinear tf, `min_df=3`, and up to 40k features feeds a multinomial logistic
regression with balanced class weights. This is a strong, fast, and
**interpretable** baseline. The linear coefficients name the exact tokens that push a
complaint toward each component, which powers the app's explanations and lets us audit
what the model learned. Class weighting counteracts the residual imbalance.

**(c) Deep learning, TextCNN with GloVe transfer learning** (`TextCNNModel`,
deployed). An embedding layer initialised with pretrained 200-d GloVe vectors
(Pennington et al., 2014, covering ~85% of our vocabulary) and fine-tuned during
training feeds four parallel 1-D convolutions of widths 2/3/4/5 with 160 filters each,
acting as learned n-gram detectors. Max-over-time pooling keeps the strongest
activation per filter, then dropout of 0.4, then a linear classifier. It is trained with Adam and
class-weighted cross-entropy, early-stopped on validation macro-F1.
The CNN learns phrase-level, order-sensitive cues like "lost power" and "goes to
the floor" that bag-of-words misses. The GloVe initialisation is a lightweight
form of transfer learning. It injects general-language knowledge the model cannot
learn from ~26k complaints alone, and that is what lifts it from clearly behind the
classical baseline up to parity with it (section 6). The pretrained vectors are needed only at
training time. The learned embedding matrix is baked into the ~15 MB `.pt`, so the
deployed model stays tiny, CPU-fast, and fully self-contained. We use GloVe rather than
fine-tuning a large transformer to keep the system lightweight and deployable, and section 9
discusses that trade-off.

### 5.3 Hyperparameter tuning strategy

We tuned on the validation split, not the test split. For the classical model the
key knobs are the TF-IDF vocabulary via `ngram_range`, `min_df`, and `max_features`, and the
LogReg regularisation `C`. We favour 1-2 grams for phrase cues like "check engine" and a
moderately low `C` for generalisation, with `class_weight="balanced"` fixed by the
imbalance. For the TextCNN, filter widths {2,3,4,5} and a count of 160 follow Kim
(2014)'s well-established design. The biggest lever we found was the embedding
initialisation. Switching from random to pretrained GloVe (200-d) added roughly two
macro-F1 points, and helped most in the low-data regime (section 7). We also probed embedding
dimension and vocabulary cut-off. An enhanced-capacity variant with mean+max pooling plus a
hidden layer, and a larger `min_freq=1` vocabulary, each slightly hurt on validation,
so we kept the simpler, better-generalising design. Dropout of 0.4, Adam at 1e-3, and
early stopping on validation macro-F1 with patience 5 are the remaining controls.
They prevent overfitting without a manual epoch search.

---

## 6. Results

### 6.1 Model comparison (held-out test set, n = 5,602)

| Model | Accuracy | **Macro-F1** | Weighted-F1 |
|---|---|---|---|
| Naive (majority class) | 0.094 | **0.012** | 0.016 |
| Classical (TF-IDF + LogReg) | 0.774 | **0.768** | 0.775 |
| Deep (TextCNN + GloVe) *(deployed)* | 0.771 | **0.762** | 0.770 |

Two things stand out. First, the naive baseline validates our metric choice. It
posts 0.094 accuracy but 0.012 macro-F1, because it gets one class fully right and the
other 13 entirely wrong. Any model has to be judged on macro-F1, and both real models
clear the floor by ~0.75. Second, the classical and deep models are a statistical
tie: 0.768 vs 0.762 macro-F1 and 0.774 vs 0.771 accuracy, a difference within run-to-run
noise. With GloVe transfer learning the neural model reaches
parity with a very strong linear baseline, but it does not surpass it. The
component signal is largely lexical, with words like "brake", "airbag", and "steering",
which is exactly what TF-IDF captures best.

![Accuracy, macro-F1, and weighted-F1 for the three models on the test set.](data/outputs/plots/model_comparison.png)

### 6.2 Per-class performance

Both models are well-balanced across the long tail. The head/tail F1 gap is only
~0.02-0.03 (section 7.4), a payoff from capping head classes and class-weighting. Per-class
F1 for the deployed deep model ranges from strong to hard:

- **Easiest:** AIR BAGS 0.91, SEATS/SEAT BELTS 0.88, EXTERIOR LIGHTING 0.87,
  VISIBILITY/WIPER 0.86. These have distinctive vocabulary like "airbag" and "seat belt".
- **Hardest:** VEHICLE SPEED CONTROL 0.53, ELECTRICAL SYSTEM 0.57, DRIVER ASSISTANCE
  (ADAS) 0.61. These are semantically diffuse or overlapping systems.

### 6.3 What gets confused (and why)

The dominant confusions, from `confusion_deep.png`, are all semantically adjacent
systems: ENGINE vs POWER TRAIN with 87 test cases in both directions, POWER TRAIN  vs 
ELECTRICAL SYSTEM at 72, STRUCTURE vs VISIBILITY/WIPER at 48, SERVICE BRAKES vs ADAS at 33,
STEERING vs SUSPENSION at 28, and ENGINE vs FUEL/PROPULSION at 28. "Electrical system" is the
worst offender because it is a genuine catch-all that co-occurs with almost everything,
as in "check engine light" or "sensor fault". These are not random errors. They mirror real
ambiguity in how a symptom maps to a subsystem, which motivates the error analysis and
the multi-label direction in future work.

![Row-normalised confusion matrix for the deployed deep model. The bright off-diagonal cells are the adjacent systems it confuses.](data/outputs/plots/confusion_deep.png)

## 7. Experiment Write-Up

### 7.1 Primary experiment: training-set-size sensitivity (the cold-start question)

**Plan.** A real deployment starts cold. When a platform begins tracking a new vehicle
or launches the model, it has few labelled complaints. How much data does each model
need? We retrain every model on stratified subsets, from 5% up to 100% of the training set, and
score macro-F1 on the fixed test set. The setup is controlled and directly relevant
to deployment economics.

**Results (macro-F1).**

| Train size | 1,306 | 2,613 | 6,534 | 13,069 | 26,138 |
|---|---|---|---|---|---|
| Naive | 0.012 | 0.012 | 0.012 | 0.012 | 0.012 |
| Classical | 0.707 | 0.726 | 0.748 | 0.762 | **0.768** |
| Deep (TextCNN + GloVe) | 0.706 | 0.732 | 0.753 | 0.762 | 0.765 |

**Interpretation.** This is the experiment where GloVe pays off. In an earlier
from-scratch version of the CNN with no pretrained embeddings, the deep model trailed the
classical model by 11 macro-F1 points at the smallest size, 0.596 vs 0.707. It
was starved for data. With GloVe transfer learning that gap disappears. At 1,306
examples the two are tied, 0.706 vs 0.707, and in the mid-data regime the deep model
actually edges ahead, 0.732 vs 0.726 at 2.6k and 0.753 vs 0.748 at 6.5k. Both curves
are still rising gently at full data, with the classical model finishing marginally on
top.

![Learning curves. With GloVe, the deep model tracks the classical model even at small training sizes.](data/outputs/plots/learning_curve.png)

**Recommendation.** Pretrained embeddings should be treated as mandatory for the neural
model in any low-resource or cold-start deployment. Without them the classical model is
strictly preferable. At full data either model is a fine choice.

### 7.2 Robustness to noisy text

Real complaints are full of typos and inconsistent casing, so we inject character-level
noise (deletion/substitution/insertion) at increasing rates and re-score.

| Char-noise rate | 0% | 5% | 10% | 20% | 30% |
|---|---|---|---|---|---|
| Classical | 0.768 | 0.724 | 0.684 | 0.555 | 0.418 |
| Deep (TextCNN + GloVe) | 0.762 | 0.704 | 0.617 | 0.431 | 0.276 |

**Interpretation.** The classical model is clearly more robust. At 10% noise it
holds 0.684 against the CNN's 0.617, and the gap widens as noise grows. The reason is that the word-level
CNN vocabulary sends every typo to `<unk>` and discards the signal, whereas TF-IDF's larger
n-gram vocabulary degrades more gracefully. This is the strongest argument for the
classical model, and for the future-work move to subword embeddings.

![Macro-F1 as character noise increases. The classical model holds up better than the word-level CNN.](data/outputs/plots/robustness.png)

### 7.3 Confidence-gated abstention (deployment triage)

Because a triage system can defer, we sweep a confidence threshold on the deployed
model: answer only above `t`, route the rest to a human.

| Threshold `t` | 0.0 | 0.5 | 0.7 | 0.9 |
|---|---|---|---|---|
| Coverage (fraction answered) | 1.00 | 0.89 | 0.77 | 0.60 |
| Accuracy on answered | 0.771 | 0.822 | 0.864 | 0.912 |

**Interpretation & recommendation.** Abstention buys large accuracy gains. Answering the
most-confident 77% of complaints reaches 86% accuracy, and the top 60% reach
**91%**. A production system should auto-route above a tuned threshold and escalate the
uncertain remainder. That turns a 0.77-accuracy model into a high-precision autorouter
plus a managed human queue.

![Accuracy against coverage as the confidence threshold rises. Answering fewer, more confident cases lifts accuracy sharply.](data/outputs/plots/confidence_coverage.png)

### 7.4 Head vs. tail

Splitting classes into the 5 most frequent head classes and 9 rarer tail classes, the
classical model scores head 0.782 and tail 0.761, while the deep model scores head 0.781 and tail 0.751. The ~0.02-0.03 gap is
small, which confirms that the head-class cap plus class-weighted training kept the
rare-but-critical systems from being neglected.

## 8. Error Analysis

We surfaced the deployed model's most confident mistakes, all at about 1.00 confidence.
These are the dangerous kind, where the model is both wrong and sure. Here are five representative cases.

1. **"noise on front passenger side suspension"**, *true: STEERING, pred: SUSPENSION.*
   Root cause: label ambiguity between two adjacent chassis systems. The narrative
   literally says "suspension" while NHTSA's primary code is steering. Mitigation:
   move to multi-label prediction, or merge the two into a "steering/suspension" super-class.
   We could also abstain on very short narratives, since this one is 6 words.

2. **"Takata inflators ... not one airbag deployed [after crash]"**, *true: STEERING,
   pred: AIR BAGS.* Root cause: a multi-component complaint whose salient text is
   entirely about airbags, but whose primary NHTSA code is steering. The model reads the
   text correctly. The single-label framing is what's wrong. Mitigation: multi-label
   output so both AIR BAGS and STEERING can be returned.

3. **"check engine light ... smelling gasoline in the cabin"**, *true: ELECTRICAL SYSTEM,
   pred: FUEL/PROPULSION.* Root cause: a strong lexical cue, "gasoline", hijacks the
   prediction over the true electrical fault. Mitigation: richer context modeling, since
   the fix was electrical, or multi-label, and down-weight single-token triggers.

4. **"t-boned ... airbags did not deploy ... door smashed in"**, *true: STRUCTURE, pred:
   AIR BAGS.* Root cause: crash narratives mention many systems, and the vivid "airbags
   did not deploy" dominates the true structural-damage label. Mitigation: add crash-context features or multi-label. These high-severity cases warrant human review anyway,
   so route them via the safety-critical tier.

5. **"front lights are extremely bright for other drivers ... glare"**, *true:
   VISIBILITY/WIPER, pred: EXTERIOR LIGHTING.* Root cause: taxonomy overlap. Glare
   is a visibility problem described entirely with lighting vocabulary.
   Mitigation: clearer annotation guidance, or a hierarchical taxonomy that lets
   "headlight glare" live under both.

**Cross-cutting theme.** The confident errors are overwhelmingly multi-component
complaints where a single primary label cannot capture the text, or genuinely
adjacent and ambiguous systems. Both point to the same highest-value fix, multi-label
classification with calibrated per-label thresholds, rather than more model capacity.

---

## 9. Commercial Viability

Is this suitable for real-world use? Partly, with the right framing. The system
is a genuine fit for a human-in-the-loop triage product, not an autonomous
decision-maker. Concretely:

- **Where it fits.** Any organisation ingesting free-text vehicle feedback faces
  the same manual-tagging bottleneck, whether that is a
  manufacturer's quality or warranty inbox, an auto marketplace like the sponsor's
  platform, a fleet operator's maintenance logs, or an insurer's claims text. A ~0.77 macro-F1 router that also abstains on
  low-confidence cases can auto-tag the confident majority and escalate the rest.
  That cuts triage cost materially while keeping a human on the hard calls.
- **Why the design is deployable.** The model is a ~15 MB artefact running CPU
  inference in milliseconds with no GPU, so it is cheap to host and scale. It is also interpretable,
  with word-level evidence and a safety tier, which is essential for adoption by safety
  reviewers who have to justify their decisions.
- **Why not fully autonomous.** Safety is high-stakes and the tail classes are the ones
  that matter most. At ~0.77 macro-F1 the model is a productivity multiplier, not a
  replacement for expert judgement. The confidence-gating result (section 7) is the mechanism
  that makes this safe: answer automatically only where the model is reliable.
- **The model-choice trade-off.** Our evaluation shows a strong classical baseline is
  on par with the neural model, so a lean production system could ship the classical
  model alone. We deploy the GloVe TextCNN to demonstrate the deep approach and because
  it generalises semantically, but a cost-sensitive buyer could run the classical model
  at a fraction of the footprint. That is an honest finding that de-risks the product.

**Verdict:** commercially viable as a triage-assist feature, not as an unsupervised
safety authority.

## 10. Ethics Statement

- **Data & privacy.** NHTSA complaints are public-domain records, but narratives can
  contain personal details. Our pipeline strips VINs, phone numbers, emails, and
  redaction artefacts before modelling, and we commit only a small sample plus a
  reproducible download script rather than re-publishing the corpus.
- **Safety-critical misuse.** A wrong or over-trusted prediction could misroute a
  genuine safety defect. The system is explicitly positioned as a triage aid with
  human review and confidence-gated abstention. It must not be used to dismiss
  complaints or to make recall decisions autonomously.
- **Bias & representativeness.** The corpus skews toward high-volume U.S. makes and models
  and toward English-language, self-selected complainants. Performance may be worse for
  under-represented vehicles, populations, or non-English text. Per-class head and tail
  reporting is our first guard against silently failing rare-but-critical systems.
- **Transparency.** Every prediction ships with the evidence behind it and a calibrated
  confidence, so reviewers can audit and overrule the model rather than defer to it.

## 11. Conclusions & Future Work

**Conclusions.** On 37k real NHTSA complaints, a disciplined three-model comparison for
14-way component routing shows four things. First, the task is very learnable from narrative text.
Both the classical TF-IDF model and the GloVe-initialised TextCNN reach ~0.77 macro-F1,
crushing the 0.01 naive floor. Second, the metric choice matters more than the model.
Accuracy alone would have hidden the naive baseline's uselessness and the tail-class
behaviour. Third, transfer learning through GloVe is what makes the neural model competitive.
It closes a two-point gap and helps most when labels are scarce. Fourth, a simple,
interpretable classical model is a remarkably strong, cheap baseline that a real product
could ship. The deployed app turns these models into a usable triage tool with
word-level explanations, safety-tier routing, and confidence-gated abstention.

**Future work, given another semester.**

- **Multi-label routing.** About 33% of complaints touch several systems, so move from
  primary-component single-label to true multi-label prediction with per-label
  thresholds.
- **Subword / transformer encoders.** Fine-tune a domain-adapted DistilBERT or use
  fastText subword embeddings, both to chase the accuracy ceiling and to fix the
  character-noise brittleness the robustness study exposed.
- **Calibration & abstention policy.** Temperature-scale the probabilities and learn a
  cost-sensitive abstention threshold per class. Missing an air-bag defect is not the same as missing a
  wiper complaint.
- **Temporal defect detection.** Layer per-component time-series anomaly detection on
  top of the router to surface emerging defect clusters, which is the actual NHTSA mission.
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
