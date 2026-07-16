# Day 9, in plain English — the builders show up

## The one line

The surveyor mapped the land, the strategist wrote the plan — and today the
**builders** arrived. One shapes the raw materials into something better to build
with; the other actually builds, tests, and hands back the crew's very first real
score.

## The two new workers

**The Feature Engineer** is the crafts-person who prepares the materials. Given the
plan, they invent a few new, useful measurements out of the ones you already have —
the kind of thing an experienced hand knows is worth having on the bench before you
start. Today's default is a simple, safe one: for each row, count how many blanks it
has (rows missing a lot of fields often behave differently). It's a small, honest
improvement that never backfires.

**The Trainer** is the builder. They take the plan and the prepared materials, try
each of the recommended approaches, grade every one fairly by cross-checking on the
practice data, keep the best, and save it. This is the first worker on the whole
assembly line who produces an actual **number**.

## The part we're most careful about: running code we didn't write by hand

Here's the interesting bit. When the AI assistant is available, we let it *write the
feature-engineering code itself* — and then we don't just trust it. We take that code
into a **sealed room** (the sandbox we built on Day 6), run it on the practice data,
and check it followed the rules: it didn't secretly throw away rows, it didn't peek at
the answer, every new measurement is actually a number. Only if it passes do we use it.
If it fails — or the AI is unavailable — we quietly fall back to our safe, built-in
default and **write down exactly what happened**.

That's the honest version of "let the AI do the clever stuff": we get the upside of a
good idea and we're completely protected from a bad one. (And true to form, the AI
provider was locked out again today — same account restriction as yesterday — so the
crew used its dependable default and said so, plainly.)

## The first real score — and what it honestly means

On the credit dataset, the builder tried three approaches and the random-forest won,
scoring about **0.80** on our fairness measure. Good news: that's a healthy number.

But we're being scrupulous about what it *is*. This is a score from **cross-checking on
the practice data** — the model grading itself by hiding parts of the practice set from
itself and testing on them. It is **not** the final-exam score. We have a sealed "final
exam" (the held-out data) that nobody in the crew is allowed to touch until the very end,
and the real head-to-head — *does the crew actually beat the lone expert and the
off-the-shelf tool?* — happens in a few days on that sealed exam. Today's milestone is
simpler and still important: **the crew now produces an honest number and a saved model,
end to end.** We refuse to dress up a practice score as an exam score.

## The honest fine print

- **Practice score, not exam score.** Every number today is cross-checking on the
  training data, labelled as such, and an automated check confirms the sealed exam is
  still untouched and unchanged after the builder finishes.
- **Guilty until validated.** Any code the AI writes has to survive a run in the sealed
  room before we'll use a line of it. A crash or a rule-break means an automatic,
  recorded fallback — never a silent bad result.
- **No extra time, no unfair advantage.** The whole build fits inside the same time
  budget we gave the off-the-shelf tool it'll eventually be compared against, so a future
  win can't be explained away as "well, you gave the crew longer."

## Where this leaves us

Four of the crew's real specialists are now working — survey, plan, prepare, build — the
first honest scores and a saved model are in hand, and all **163** automated checks are
green. The line now takes a dataset and produces a trained model, entirely on its own.

Tomorrow (Day 10) the **Critic** joins: the reviewer who looks at the builder's result,
diagnoses what actually went wrong, and sends the strategist back a specific fix. That's
the day the crew stops being a straight line and becomes a team that *learns from its own
first attempt*.
