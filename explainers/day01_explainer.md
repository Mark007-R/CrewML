# Day 1, explained: why we built the referee before the players

Most ML projects start with the model. CrewML starts with the *referee*, and
that choice is the whole point of Day 1.

The project makes a bold promise: a **team of AI agents** that argue with each
other about how to model your data will beat a **single AI agent** doing it in one
shot — and hold its own against a serious classical AutoML tool. Promises like that
are easy to make and, in a lot of ML demos, quietly cheated. The cheating is almost
never dramatic; it's mundane. You tune your model while peeking at the test scores.
You scale your features using statistics computed over the whole dataset, test rows
included. By the time you report a number, the "unseen" data has leaked into your
model a dozen small ways. The result looks great and means nothing.

So before writing a single agent, we built the thing that makes cheating hard.

### Locking the test away

We picked five public datasets — some where the model predicts a yes/no answer
(will this loan go bad?), one where it picks among four vehicle types, and two where
it predicts a number. For each, we split the data **once** into a training pile and
a held-out pile, then **locked the held-out pile away**. The crew, when we build it,
will only ever be handed the training pile. The held-out pile comes out exactly
once, at the very end, to produce the score that goes in the report.

To make "locked" mean something enforceable, we took a **cryptographic fingerprint**
(a SHA-256 hash) of the held-out data and wrote it into a manifest we commit to git.
A hash is a short string that changes completely if even one number in the data is
altered. Later, a single function call re-fingerprints the held-out set and compares
it to the recorded one. If they don't match, someone touched the sacred data, and
the whole run is flagged as untrustworthy. We even wrote a test that *deliberately*
tampers with the data to confirm the alarm actually goes off — a smoke detector is
only reassuring if you've held a match under it.

### Choosing the right yardstick per dataset

A subtler kind of dishonesty is measuring with the wrong ruler. For the imbalanced
loan dataset — where 70% of cases are "good" — a model that blindly says "good"
every time is right 70% of the time and completely useless. So we don't score it on
plain accuracy. We use **ROC AUC**, which measures whether the model ranks risky
loans above safe ones regardless of where you draw the line. For the four-way
vehicle problem we use **macro-F1**, which forces the model to do well on *every*
class, not just the common ones. For the number-predicting datasets we use **R²**,
how much of the variation the model actually explains. Each dataset gets exactly one
official yardstick, decided today, so nobody can shop for a flattering metric later.

### Why nothing is committed except the recipe

The datasets themselves aren't stored in the repository — only the *recipe* to
rebuild them (which datasets, which versions, which split seed) plus the
fingerprints. Anyone can run one script and reproduce byte-identical splits. This
keeps the repo lean and, more importantly, means the held-out data lives on the
machine doing the work, not floating around in version control where it could drift.

### The honest part about honesty

One more guardrail: the system can run without any AI at all, in a "mock mode," so
the plumbing can be tested offline. Mock mode produces fake, meaningless scores by
design — so the protocol flatly forbids ever reporting a mock number as if it were
real. If a report shows mock results, it has to shout **MOCK** in the label.

That's Day 1: no agents yet, no models, no clever loop. Just an evaluation setup
rigged so that when the crew finally does beat the solo agent, the number will be
worth believing. The exciting parts come next — but they'd be worthless without the
boring referee we built first.
