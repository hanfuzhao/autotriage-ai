# AutoTriage AI, 5-minute pitch script

This is a word-for-word script written for a calm, slightly slow speaking pace, roughly
110 words a minute. Pause where you see the slash marks. The live demo is the center of
the talk, so keep the talking short there and let the screen do the work.

Live demo: https://autotriage-ai-unfvnsiy6a-uc.a.run.app
Code: https://github.com/hanfuzhao/autotriage-ai

---

## 0:00 to 0:40, the problem

Every year, drivers in the U.S. file hundreds of thousands of safety complaints with
NHTSA. / Things like, the brakes went to the floor, or, the airbag light won't turn off,
or, the car braked by itself on the highway. / These complaints are how safety problems
get noticed early. When enough of them cluster together, they trigger investigations and
recalls. / But before anyone can study them, every single complaint has to be tagged with
the car system it is about. / Today that tagging is mostly done by hand. It is slow, and
it is inconsistent, and it backs up at exactly the wrong time.

## 0:40 to 1:20, what we built

So we built AutoTriage AI. / You give it the owner's own words, and it predicts which of
fourteen vehicle systems the complaint belongs to, in under a second. / It also flags how
safety-critical that system is, and it suggests where to route the report. / We trained it
on about thirty-seven thousand real NHTSA complaints across fourteen car makes. / And it
is not a black box. It shows you the exact words that drove the decision, so a human
reviewer can trust it, or overrule it.

## 1:20 to 3:00, the live demo

Let me show you. /

Paste a real, messy complaint. You get the predicted system, a confidence score, and the
top three guesses. /

Look at the highlighted words. Lost power, check engine. That is the model's reasoning,
right there on the screen. /

Here is the triage banner. Brakes and airbags come up as critical, with a routing action
attached. /

Now watch this. I flip between the three models on the same text. The naive baseline, the
classical model, and the deep model. You can see them agree or disagree live. /

And one hard case. Phantom braking on an empty road. It correctly lands on driver
assistance, the ADAS system.

## 3:00 to 4:10, results and the key finding

We compared three models honestly, on a long-tailed problem where plain accuracy lies. /
So we report macro-F1, which weights every safety system equally. / The naive baseline
scores almost zero, about one percent. / The classical model and the deep model both reach
about seventy-six to seventy-seven percent. /

Here is the interesting part. / Our main experiment varied how much training data each
model got. / When data is scarce, at the cold start, the deep model used to fall far
behind, by about eleven points. / Adding pretrained GloVe word vectors closed that gap
completely. With almost no data, it now ties the classical model. /

And for deployment, we let the model stay quiet when it is unsure. If it only answers its
most confident sixty percent of cases, accuracy climbs above ninety percent, and the rest
go to a human.

## 4:10 to 5:00, why it matters, and the ask

This turns manual triage into real-time triage. / The same engine works for a car
marketplace, a manufacturer's quality inbox, or a fleet's maintenance logs. / It is live
right now on the cloud, it runs on a small model with no GPU, and it is transparent enough
to put in front of a safety reviewer today. /

The ask is simple. Give us access to a partner's complaint stream, and let us run a pilot
with one quality team. / Thank you.

---

## Backup facts for questions

- Data: NHTSA ODI Complaints API, public domain. About 77 thousand crawled, 37 thousand after cleaning.
- Deployed model: TextCNN with GloVe transfer learning, about 15 MB, runs on CPU.
- Metrics: naive 0.012 macro-F1, classical 0.768, deep 0.762. Accuracy is near 0.77 for both real models.
- Hardest confusions: engine, powertrain, and fuel or propulsion, which overlap in how people describe them.
- This is a triage aid with a human in the loop, not safety advice.
