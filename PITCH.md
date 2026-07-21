# AutoTriage AI, 5-minute pitch script

Word-for-word, written to be spoken out loud at a calm pace, roughly 110 words a minute.
Slash marks are breathing points. Slide cues in brackets tell you when to advance.

Live demo: https://autotriage-ai-unfvnsiy6a-uc.a.run.app
Code: https://github.com/hanfuzhao/autotriage-ai

**Before you start.** Open the demo and run one query a minute beforehand. The service sleeps
when idle and the first request takes ten to fifteen seconds to wake, which is dead air you do
not want on the clock. Have both example complaints already copied. The spoken text below is
about 470 words, roughly four and a quarter minutes, leaving slack for anything that loads slowly.

---

## [SLIDE 1, title] 0:00 to 0:25, the hook

Think about the last time your car did something a little off. A noise, a warning light. /
Could you tell whether that was dangerous, or nothing? / Most of us can't. So we either worry
over nothing, or we keep driving through something real.

## [SLIDE 2, the problem] 0:25 to 0:55, why it matters

Here's what most people don't know. Safety recalls start with ordinary owners filing complaints
with NHTSA. / But only if the owner recognises there's a problem and files it against the right
system. / That gap is what I went after.

## [SLIDE 3, what I built] 0:55 to 1:40, the product, and why it's new

So I built AutoTriage. You describe what your car did, in plain words. / It tells you which of
fourteen car systems it's about, how serious that is, and whether it's worth reporting. Trained
on thirty-seven thousand real NHTSA complaints, and it shows the words it keyed on. /

Two things make this new. / I did car text for the hackathon, but that was sentiment, how an
owner feels about a review. Here it's safety complaints, not reviews. Which system failed, not
how someone feels. A TextCNN, not a fine-tuned transformer. / And the published work on this
database clusters it for regulators. I pointed it at the owner instead.

## [SLIDE 4, live demo] 1:40 to 2:55, the demo

Let me show you. /

Airbag light stays on, dealer says the sensor is bad. Paste it in. / Air bags, ninety-something
percent, flagged critical, with a next step: get it checked, worth reporting. /

Look at the highlighted words. Airbag, sensor, deploy. That's the reasoning, on screen. /

Now a flickering headlight. / Lower tier, keep-an-eye-on-it. / And I can flip between all three
models on the same text.

## [SLIDES 5 and 6, models and results] 2:55 to 3:55, how I judged it

Three models, judged honestly. / This data is lopsided: common systems have thousands of
complaints, rare ones a few hundred. / So I don't report plain accuracy, because a lazy model
looks strong while missing the rare systems. I report macro-F1, which weights every system
equally. / The naive baseline scores about one percent. Both real models land near seventy-seven. /

That choice came straight from the hackathon, where accuracy read eighty-six percent while the
model caught almost none of the unhappy customers. I wasn't going to be fooled twice.

## [SLIDE 7, what I learned] 3:55 to 4:25, the key finding

The finding I'm proudest of. / When training data is scarce, my neural model used to fall about
eleven points behind the simple one. Adding pretrained word vectors closed that gap completely. /
And for a safety tool, this matters most: when the model is unsure, it says so, instead of
handing someone a false all-clear.

## [SLIDE 8, why it matters and the ask] 4:25 to 4:50, the close

Take that helpless feeling and turn it into a clear read and a next step. / And because every
report an owner files feeds the recall system, helping one driver helps the next. / It's live
now, a tiny model, no GPU, and it explains itself. /

The ask: put this in front of a car marketplace or an insurer's claims flow, with real owners. /
Thank you.

---

## Backup facts for questions

- **Is this your hackathon project?** No. The hackathon was per-aspect sentiment on Edmunds consumer reviews with a fine-tuned DistilBERT. This is 14-class component routing on NHTSA safety complaints with a TextCNN and GloVe. Different data, task, model, and user. What carried over was the evaluation lesson, not the code.
- Data: NHTSA ODI Complaints API, public domain. About 77 thousand crawled, 37 thousand after cleaning. Split 70/15/15, so 26,138 train, 5,602 validation, 5,602 test.
- Deployed model: TextCNN with GloVe transfer learning, about 15 MB, runs on CPU.
- Metrics: naive 0.012 macro-F1, classical 0.768, deep 0.762. Accuracy near 0.77 for both real models.
- Hyperparameter search: grid over the classical C and n-gram range, and the TextCNN dropout and filter count, selected on validation only. The surface is flat, so the shipped config sits within noise of the grid winner.
- Prior work: Ghazizadeh, McDonald and Lee (2014) text-mined this same database, but by clustering for defect discovery, not supervised routing for owners.
- Hardest confusions: engine, powertrain, and fuel or propulsion, which overlap in how people describe them.
- Safety framing: a first read for owners, not a diagnosis, and no substitute for a mechanic.
