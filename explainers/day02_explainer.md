# Day 2, in plain English — "How good is good?"

Yesterday we locked five datasets and hid a slice of each one away in a sealed
vault (the *holdout*) so nobody can cheat by studying the answer key. Today we
answered a simple but essential question: **before we build anything clever, how
well can we do with almost no effort?** You need that number, because "our fancy
crew got 0.78" means nothing until you know whether 0.78 is impressive or
embarrassing.

So we built two deliberately dumb reference points.

**The floor: a model that ignores the data.** Imagine a weather forecaster who
never looks out the window and just says "same as usual" every day. That's our
"Dummy" model — it makes the same guess no matter what the input is. For the
yes/no datasets it scored **exactly 0.5**, which is the score you'd get flipping a
coin. For the number-prediction datasets it scored **about 0**. This is the floor:
anything that can't beat this is broken, full stop.

**The anchor: throw a standard tool at it, untouched.** Then we took a
RandomForest — a solid, popular, off-the-shelf model — and ran it with zero tuning
and only the bare-minimum data cleaning needed to make it run. Think of it as "what
a competent person gets in five minutes without trying hard." It scored far above
the floor on every dataset:

| Dataset   | Coin-flip floor | Five-minute forest |
|-----------|----------------:|-------------------:|
| credit-g  |            0.50 |               0.78 |
| diabetes  |            0.50 |               0.81 |
| vehicle   |            0.10 |               0.73 |
| cpu_small |            0.00 |               0.97 |
| kin8nm    |            0.00 |               0.69 |

(Higher is better in every row.)

**Why this matters for the whole project.** The entire promise of CrewML is that a
*team* of AI agents that critique each other beats a *single* AI agent working
alone — and beats classical automated tools too. To prove that honestly, you need
a scoreboard everyone plays on. Today we built that scoreboard (one shared scoring
module) and filled in the first two rows. Over the next two days we add the solo
agent and the classical AutoML tool. Only then can the crew step up and try to
beat all of them — on data it has never once been allowed to see.

One quiet but important detail: after scoring, we re-checked the vault's tamper
seal on all five datasets. Untouched. The honesty guarantee still holds.
