# Day 6, in plain English — the one power tool the whole crew shares

## The one line

Our AI agents don't just *talk* about machine learning — they **write real Python
and run it on the data**. Today we built the single tool that runs that code: a
walled-off workspace with a stopwatch, a clipboard for the results, and a lock on
the door. Every worker we hire from here on reaches for this one tool.

## Why this is the most important piece

Yesterday we built the empty assembly line. But an assembly line is useless until
there's a machine the workers actually operate. For this crew, that machine is a
**code runner**. The Feature Engineer will write code to reshape the data; the
Trainer will write code to build a model. None of that matters unless something can
take a block of freshly-written, possibly-buggy code and *run it* — safely, and
report back what happened.

That "safely, and report back" is the entire job of today's tool. Get it wrong and
one bad line of generated code hangs the whole system, or worse, quietly reads
something it shouldn't.

## What the tool actually does

Think of it as handing a contractor a **sealed room** to work in:

- **Its own room, every time.** Each run gets a fresh, empty workspace. We carry in
  exactly the files it's allowed to see — nothing more — and whatever it produces
  stays in that room. Two jobs never step on each other.
- **A stopwatch on the door.** The code gets a time limit (two minutes by default).
  If it gets stuck in an infinite loop, the tool doesn't wait forever — it opens the
  door, ends the job, and reports "timed out." We proved this by actually feeding it
  an infinite loop and watching it get stopped at the 2-second mark.
- **A clipboard for results.** The code hands back two things through a little
  drop-box: a sheet of **numbers** (like "this model scored 0.76") and any **files**
  it made (like the trained model itself). The tool collects both and returns them
  in a tidy package.
- **Crashes are news, not disasters.** If the code blows up, the tool doesn't blow
  up with it — it calmly reports "this failed, here's the error." That matters
  enormously, because on Day 20 the crew learns to *read those errors and fix its
  own code*. You can't recover from a crash you never see.

## Seeing it work

We ran a realistic dry run: hand the tool a block of code that trains a model on
the (training-only) credit data. It ran the code, came back with a score of
**0.76**, and a saved model file — exactly the round-trip the real Trainer will do
next week. Then we deliberately fed it a crash and an infinite loop, and it handled
both without flinching.

## The honest fine print

Two things we're careful to be straight about:

- **This is a "safe workspace," not yet a "bank vault."** It isolates code and
  stops runaways — which is all we need while the crew is running *its own* code.
  The heavy-duty security locks (no internet access, a strict list of what the code
  is allowed to import) come on **Day 19**, when we deliberately try to break it. We
  wrote that limitation right into the code's own documentation so no future
  write-up can oversell it.
- **It still can't peek at the final exam.** The tool only ever sees the files we
  personally carry into the room, and it knows nothing about our locked test set. A
  automated check even scans its code and **fails the build** if the forbidden word
  so much as appears. No peeking — enforced by structure, not by good intentions.

## Where this leaves us

The power tool works, all **105** automated checks are green, and the crew now has
the one thing it truly can't function without. Everything from here is hiring the
actual workers who'll use it.

Tomorrow (Day 7) we hire the first real one: the **Profiler** — the specialist who
opens up the dataset and writes down what's actually in it (what the columns are,
what's missing, how the thing we're predicting is distributed) before anyone tries
to model it.
