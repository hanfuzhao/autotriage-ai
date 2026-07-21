# AutoTriage AI, 5-minute pitch script

Word-for-word script written for a calm, slightly slow pace, roughly 110 words a minute.
Pause where you see the slash marks. The live demo is the centre of the talk, so keep the
talking short there and let the screen do the work.

Live demo: https://autotriage-ai-unfvnsiy6a-uc.a.run.app
Code: https://github.com/hanfuzhao/autotriage-ai

**Before you start.** Open the demo and run one query a minute beforehand. The service sleeps
when idle and the first request takes ten to fifteen seconds to wake it, which is dead air you
do not want on the clock. Have the two example complaints already copied. The spoken script
below is about 430 words, which is roughly four minutes at this pace, leaving a full minute of
slack for the demo. Treat that slack as the budget for anything that loads slowly.

---

## 0:00 to 0:40, the problem

Think about the last time your car did something a little off. / A noise you had not heard
before. A warning light. A moment where it just did not feel right. / Could you tell if that
was dangerous, or nothing? / Most of us can't. So we either worry over nothing, or we keep
driving through something that turns out to be a real defect. / And here is the part people
do not know. Most recalls start with ordinary owners filing complaints with NHTSA, but only
if they realize there is a problem and file it against the right system.

## 0:40 to 1:20, what I built

So I built AutoTriage. / You describe what your car did, in plain words, the way you would
tell a friend. / It tells you three things: which of fourteen car systems it is about, how
serious that system usually is, and whether it is worth reporting. / It learned from
thirty-seven thousand real NHTSA complaints. / And it does not just hand you a label. It
shows the exact words it keyed on, so you can decide whether to trust it.

## 1:20 to 2:50, the live demo

Let me show you. /

Here is a scary one. The airbag light stays on and the dealer says the sensor is bad. /
Paste it in. / Air bags, ninety-something percent sure, flagged critical, with a plain next
step: get it checked, and this is worth reporting. /

Look at the highlighted words. Airbag, sensor, deploy. That is why it decided. /

Now a calmer one, a flickering headlight. / Lower tier, more of a keep-an-eye-on-it. /

And I can flip between all three models on the same text, to watch them agree or disagree.

## 2:50 to 4:00, results and the key finding

I compared three models honestly. / This data is lopsided: common systems have thousands of
complaints, rare ones only a few hundred. / So I do not report plain accuracy, because a lazy
model can look good while missing the rare systems. I report macro-F1, which weights every
system equally. / The naive baseline scores about one percent. Both real models land near
seventy-seven. /

The finding I am proudest of. / When training data is scarce, my neural model used to fall
about eleven points behind. Adding pretrained word vectors closed that gap completely. /

And for a safety tool this matters most: when the model is unsure, it says so, instead of
handing you a false all-clear.

## 4:00 to 4:40, why it matters, and the ask

Take that helpless feeling when your car acts up, and turn it into a clear, honest read and a
next step. / And because every good report an owner files feeds the recall system, helping one
worried driver helps the next. / It is live right now, runs on a tiny model with no GPU, and
it explains itself. /

The ask: let me put this in front of a car marketplace or an insurer's claims flow, and run it
with real owners. / Thank you.

---

## Backup facts for questions

- Data: NHTSA ODI Complaints API, public domain. About 77 thousand crawled, 37 thousand after cleaning.
- Deployed model: TextCNN with GloVe transfer learning, about 15 MB, runs on CPU.
- Metrics: naive 0.012 macro-F1, classical 0.768, deep 0.762. Accuracy is near 0.77 for both real models.
- Hyperparameter search: grid over the classical C and n-gram range and the TextCNN dropout and filter count, selected on validation only. The response surface is flat, so the shipped configuration sits within noise of the grid winner.
- Novelty: prior work on this database, notably Ghazizadeh, McDonald and Lee (2014), clusters the archive for regulators. This is supervised routing pointed at the owner instead.
- Hardest confusions: engine, powertrain, and fuel or propulsion, which overlap in how people describe them.
- Safety framing: a first read for owners, not a diagnosis, and no substitute for a mechanic.
