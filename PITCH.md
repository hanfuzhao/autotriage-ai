# AutoTriage AI — 5-Minute Pitch

> Investor-demo framing. Target: 5:00 hard stop. Live demo is the centrepiece.

---

### 0:00 – 0:45 · Problem & motivation

Every year, U.S. drivers file **hundreds of thousands** of safety complaints with
NHTSA — "the brakes went to the floor", "the airbag light won't turn off", "it braked
by itself on the highway". These complaints are the *early-warning system* for auto
safety: clustered together, they trigger investigations and recalls.

But before anyone can analyze them, every complaint has to be tagged with the vehicle
system it's about. Today that's done **by hand** — slow, inconsistent, and backlogged
exactly when a cluster of brake failures needs to be caught *fast*. Any platform that
takes in owner feedback about cars has this same triage problem.

### 0:45 – 1:30 · What we built

**AutoTriage AI** reads the owner's own words and routes the complaint to the right one
of **14 vehicle systems** in a fraction of a second — then flags how safety-critical it
is and suggests where to route it. It's trained on **37,000 real NHTSA complaints**
across 14 makes. Crucially, it's not a black box: it shows you the exact words that
drove the decision, so a human reviewer can trust — or overrule — it.

### 1:30 – 3:15 · Live demo  ⟵ *the core*

1. Paste a messy real complaint → instant prediction, confidence ring, top-3 systems.
2. Point at the **highlighted words** — "lost power", "check engine" — the model's
   reasoning, visible.
3. Show the **triage banner**: brakes/air-bags come up *critical* with a routing action.
4. Flip the **model selector** (Deep → Classical → Naive) on the same text to show the
   three-tier comparison live.
5. Try an adversarial one (phantom braking) to show ADAS handling.

### 3:15 – 4:15 · Results & key findings

- Three models, evaluated honestly on a **long-tailed** 14-class problem where accuracy
  lies. We report **macro‑F1** so rare-but-critical systems count as much as common ones.
- Both real models hit **~0.77 macro-F1** vs a **0.01** naive floor. Deployed TextCNN:
  0.762 macro-F1 / 0.771 accuracy; classical baseline essentially tied at 0.768 / 0.774.
- **Key insight from our experiment:** a training-set-size study shows the "cold-start"
  curve a safety team faces when a new model launches. Pretrained **GloVe transfer
  learning** erases the neural model's 11-point low-data deficit — at ~1,300 examples it
  goes from far behind the classical model to dead even.
- **Built for deployment:** confidence-gated abstention lets the system answer the easy
  cases automatically and hand the uncertain ones to a human.

### 4:15 – 5:00 · Why it matters / ask

Manual triage → real-time triage. The same engine works for a manufacturer's quality
inbox, an auto marketplace, or a fleet's maintenance logs. It's live, it's self-contained
(a few-MB model, no GPU), and it's interpretable enough to put in front of a safety
reviewer today. **The ask:** [partner data access / pilot with a QA team / next-quarter
roadmap].

---

**Backup Q&A facts**

- Data: NHTSA ODI Complaints API, public domain; 77.5k crawled → 37.3k after cleaning.
- Deployed model: TextCNN (Kim 2014) with GloVe transfer learning, CPU inference, ~15 MB.
- Hardest confusions: engine ↔ powertrain ↔ fuel/propulsion (semantically adjacent).
- Not safety advice; a triage aid with a human in the loop.
