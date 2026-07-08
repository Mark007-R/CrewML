# Day 3, in plain English — the "one smart person, one try" benchmark

## The question we're really answering

The whole project rests on one promise: *a **team** of AI agents that critique each
other builds better models than a **single** AI agent working alone.* To prove that,
we first have to measure the single agent. That's today.

Think of it like hiring. Before you argue that a whole engineering team is worth
the payroll, you'd want to know how far *one* strong engineer gets on the same
problem with one attempt. If the team barely edges out the solo hire, the extra
seats aren't paying for themselves. Today we sat that solo engineer down.

## What "the solo agent" actually does

We hand one AI a short briefing about the **training data only** — how many rows,
which columns are numbers vs categories, how the answer is distributed, where
values are missing — plus the task ("predict this, and you're graded on that
metric"). In one shot, it writes a complete Python program that trains a model. We
then run its program and grade the result on the **locked test set** it was never
allowed to see.

Crucially, the AI only ever touches the practice data. When it's time to grade, a
separate, trusted piece of *our* code — not the AI's — takes the AI's trained model
and shows it the test questions (the features), collects its answers, and compares
them to the hidden answer key. The AI never sees the answer key, and nothing it
wrote is ever allowed to study the test. After grading we re-check a cryptographic
"seal" on the test set to prove not a single row was disturbed. That separation is
the entire integrity story of this project, and today we built the machinery that
enforces it for agents.

## A wrinkle: we don't have an AI key plugged in yet

Running a live model costs money and needs an API key, which isn't configured
here. So today's run is in **mock mode**: instead of a live AI, we used a fixed,
sensible "what a decent one-shot answer looks like" script. Every number from a
mock run is stamped **MOCK** and is explicitly *not* our official result — that's a
hard rule in our evaluation protocol, because reporting a fake number as real would
quietly poison the headline claim. The value of today isn't the numbers; it's that
the pipeline — brief the agent, run its code safely, grade it honestly — now works
end to end. Swap in a real key later, re-run the same command, and out come the
real numbers.

## What the (mock) numbers showed

The solo attempt beat the "always guess the average" floor on all five datasets,
and went toe-to-toe with yesterday's plain RandomForest: it pulled clearly ahead on
two datasets (one by a lot), and fell a little short on the two small yes/no
datasets — the ones with class imbalance and sneaky missing values. That's a very
realistic portrait of a smart-but-uncoached solver: strong instincts, but nobody
looking over its shoulder saying "you forgot about the imbalance." Which is
precisely the gap the crew's critic is supposed to close.

## Why this matters for what's coming

We now have a concrete, honest target line drawn on the field. When the multi-agent
crew comes online in Phase 2, "did the team beat the solo player?" won't be a vibe —
it'll be a number-versus-number on data neither of them ever saw. Tomorrow we raise
the bar one more notch with classical AutoML (a heavily-optimized non-AI tool), so
the crew has to beat not just a lone agent but a serious automated competitor too.
