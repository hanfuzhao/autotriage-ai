# AutoTriage AI, 5-minute pitch script

Word-for-word script written for a calm, slightly slow pace, roughly 110 words a minute.
Pause where you see the slash marks. The live demo is the center of the talk, so keep the
talking short there and let the screen do the work.

Live demo: https://autotriage-ai-unfvnsiy6a-uc.a.run.app
Code: https://github.com/hanfuzhao/autotriage-ai

---

## 0:00 to 0:45, the problem

Think about the last time your car did something a little off. / A noise you had not heard
before. A warning light. A moment where it just did not feel right. / Now be honest, could
you tell if that was dangerous, or nothing? / Most of us can't. So we either worry over
nothing, or, worse, we keep driving through something that turns out to be a real defect. /
And here is the part people do not know. The way most recalls actually start is ordinary
owners filing complaints with NHTSA. But only if they realize there is a problem, and file
it against the right system.

## 0:45 to 1:30, what we built

So I built AutoTriage. / You describe what your car did, in plain words, the way you would
tell a friend. / It tells you three things. Which of fourteen car systems it is about. How
serious that system usually is. And whether it is the kind of thing worth reporting. / It
learned from thirty-seven thousand real NHTSA complaints. / And it does not just hand you a
label. It shows you the exact words it keyed on, so you can decide whether to trust it.

## 1:30 to 3:00, the live demo

Let me show you. /

Here is a scary one. The airbag light stays on and the dealer says the sensor is bad. /
Paste it in. / It comes back with air bags, ninety-something percent sure, and it flags it
as critical, with a plain next step: get it checked, and this is worth reporting. /

Look at the highlighted words. Airbag, sensor, deploy. That is why it decided. /

Now a calmer one, a flickering headlight. / Same tool, but now it is a lower tier, more of
a keep-an-eye-on-it. /

And you can flip between three models on the same text, to see them agree or disagree.

## 3:00 to 4:10, results and the key finding

I compared three models honestly. / The catch is that this data is lopsided. Common systems
have thousands of complaints, rare ones only a few hundred. / So I do not report plain
accuracy, because a lazy model can look good and still miss the rare systems. I report
macro-F1, which weights every system equally. / The naive baseline scores about one
percent. The real models, classical and deep, both land around seventy-six to seventy-seven
percent. /

The finding I am proudest of. / When training data is scarce, my neural model used to fall
far behind, by about eleven points. / Adding pretrained word vectors closed that gap
completely. /

And for a tool people trust with safety, this matters most. When the model is not sure, it
says so, instead of handing you a false all-clear.

## 4:10 to 5:00, why it matters, and the ask

The goal is simple. Take that helpless feeling when your car acts up, and turn it into a
clear, honest read, and a next step. / And because every good report an owner files feeds
the recall system, helping one worried driver also helps the next. / It is live right now,
it runs on a tiny model with no GPU, and it explains itself. /

The ask. Let me put this in front of a car marketplace or an insurer's claims flow, and run
it with real owners. / Thank you.

---

## Backup facts for questions

- Data: NHTSA ODI Complaints API, public domain. About 77 thousand crawled, 37 thousand after cleaning.
- Deployed model: TextCNN with GloVe transfer learning, about 15 MB, runs on CPU.
- Metrics: naive 0.012 macro-F1, classical 0.768, deep 0.762. Accuracy is near 0.77 for both real models.
- Hardest confusions: engine, powertrain, and fuel or propulsion, which overlap in how people describe them.
- Safety framing: a first read for owners, not a diagnosis, and no substitute for a mechanic.
