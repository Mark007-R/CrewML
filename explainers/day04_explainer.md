# Day 4, in plain English — the strongest robot competitor, and the final scoreboard

## The one line

Yesterday we measured *one* smart AI working alone. Today we brought in the
heavyweight non-AI competitor — an automated tool that tries hundreds of models by
itself — so the crew has something genuinely hard to beat. Then we put everyone on a
single scoreboard and closed out Phase 1.

## What "classical AutoML" is

Imagine a tireless robot that already knows every standard modeling recipe. You hand
it the training data and say "you have two minutes — try as many models and settings
as you can and keep the best one." It doesn't reason or write code the way an AI
agent does; it just searches, fast and methodically. That's **FLAML**, the tool we
used. It's the kind of thing a strong data scientist reaches for when they want a
solid model without hand-tuning. If our AI *crew* can't beat this, the crew isn't
worth much — so it's the perfect bar.

We gave FLAML exactly the same deal as everyone else: it only ever sees the practice
data, it's graded once on the locked test set it never saw, and it's scored by the
same yardstick. We even gave it *at least as much compute time as the crew will
get*, so if the crew wins later, nobody can say it only won because it was handed a
bigger budget.

## What happened — including a genuinely useful surprise

FLAML came out on top on three of the five datasets, and on one of them (a robot-arm
problem) it won by a mile. No shock there — that's what a good automated searcher
should do.

The surprise is the other two. On the two small yes/no datasets — a credit-risk one
and a diabetes one — the fancy AutoML tool **did not win**. On the credit dataset it
actually did a bit *worse* than yesterday's plain, untuned "random forest" model.
Why? When you have only a few hundred rows and you let an aggressive tool try
hundreds of tweaks, it can talk itself into a clever-looking setup that turns out to
be slightly *too* clever — it looked great on the practice data but generalized a
hair worse. The boring forest just quietly did its job.

That's not a failure of the experiment; it's the most valuable thing we learned
today. It tells us *where* the crew has a real shot: not on the big clean datasets
where the robot searcher already dominates, but on the small, messy, imbalanced ones
where brute-force search stumbles and a bit of actual reasoning — "hey, this data is
imbalanced, handle that" — could pull ahead. The crew's job just got a specific
address.

## The final scoreboard for Phase 1

We stacked every competitor onto one table:

| Dataset   | Always-guess floor | Plain forest | Solo AI (mock) | AutoML robot |
|-----------|-------------------:|-------------:|---------------:|-------------:|
| credit-g  |             0.5000 |   **0.7783** |         0.7521 |       0.7352 |
| diabetes  |             0.5000 |   **0.8118** |         0.7987 |       0.8039 |
| vehicle   |             0.1028 |       0.7260 |         0.7763 |   **0.7785** |
| cpu_small |            −0.0029 |       0.9726 |         0.9747 |   **0.9759** |
| kin8nm    |            −0.0002 |       0.6948 |         0.8120 |   **0.8421** |

Bold = the number to beat on that row. (The solo-AI column is greyed out in spirit —
it's still "mock", a stand-in, because we haven't plugged in a paid AI key yet, so we
never treat it as a real result.)

## Why this closes Phase 1 cleanly

Four days in, we haven't built a single crew agent yet — and that's on purpose.
You don't build the contestant before you build the scoreboard and the referee.
We now have: a locked, tamper-proof test set (checked and still untouched after
every single scoring run), one honest yardstick everybody is measured by, four
reference competitors from "coin flip" up to "serious automated tool", and a written
rulebook that forbids cheating and forbids passing off fake numbers as real. Every
piece is tested — 77 automated checks, all green.

Tomorrow, Phase 2 begins and the actual crew starts taking shape: we lay down the
skeleton of the multi-agent assembly line — the stations (profiler, planner, feature
engineer, trainer, critic) and the conveyor belt between them — before wiring each
station live. The scoreboard is set; now we build the team that has to climb it.
