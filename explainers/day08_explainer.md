# Day 8, in plain English — the strategist reads the survey

## The one line

Yesterday the surveyor walked the plot and wrote down what's there. Today we hired
the **strategist**: the one who reads that survey and decides *how* to attack the
problem — what to throw out, how to tidy up the rest, which kinds of models to try,
and how to grade them fairly — all *before* anyone builds anything.

## Why this worker comes second

A good survey is useless if nobody turns it into a plan. The **Planner** is the
person who sits down with the surveyor's notes and says: "These three columns are
junk, drop them. Those categories need tidying this way. Try these three model
families, in this order. And here's how we'll test them so we don't fool ourselves."
Everything the builders do next follows this plan.

The important rule — the same one we hold everyone to — is that the Planner works
**entirely from the survey**, and never re-opens the data itself. It's one more step
removed from any chance of cheating: it reads a summary and reasons about it, full
stop.

## What the plan actually says

For each dataset the Planner writes a short, concrete game plan:

- **What to throw out.** Columns that never change, columns that are secretly just an
  ID number, exact duplicates, or anything the surveyor flagged as "gives away the
  answer." Each removal comes with a one-line reason — no silent deletions.
- **How to tidy up.** Fill in gaps sensibly (median for numbers, most-common for
  categories). Where the surveyor spotted **fake zeros** — like the diabetes insulin
  reading that's zero when it really means "not measured" — the plan says to treat
  those zeros as gaps first. Categories with *tons* of distinct values get a compact
  encoding so we don't accidentally create a thousand new columns.
- **Which models to try.** Three families per problem, strongest-first, each with a
  small set of settings to sweep. It even notes which ones need the data rescaled and
  which can handle lopsided classes — practical hints for the builder.
- **How to grade fairly.** The right kind of cross-checking for the job (keeping class
  proportions balanced for yes/no problems), the right scorecard, and a fixed random
  seed so the same plan always tests the same way.
- **What to do about lopsidedness.** On the credit data — about 2.3 "good" for every
  "bad" — the plan switches on balancing so the model can't win by always guessing
  "good." On the datasets that *aren't* lopsided, it deliberately leaves that off.
  The strategy only kicks in when the survey earns it.

## The part we're proud of: it already knows how to take feedback

Later in the project there'll be a **Critic** who reviews the first model and says
things like "this is overfitting" or "you missed a leak." We built the Planner's
*response* to that feedback **today**, ahead of the Critic itself: tell it "overfit"
and it genuinely dials the models back and turns up the safeguards — not just a note
saying it heard you, an actual change to the plan. So when the Critic arrives, the
feedback loop already works instead of needing a second hookup.

## A smart assistant, kept on the same short leash

Like the surveyor, the Planner has an **AI assistant** meant to add a few
dataset-specific tips for the builders. Today we asked for them and the provider was
**temporarily locked out** (an account restriction on their end, nothing to do with
our data). Here's the point worth making: **nothing broke.** The plan is written by
plain rules, so it came out complete and correct with the AI's note simply marked
"unavailable," honestly, with the reason attached. We did **not** invent advice to
paper over the gap. The assistant is a bonus; the plan never depends on it.

## The honest fine print

- **"Suspected" stays "suspected."** The Planner inherits the surveyor's caution about
  fake zeros and keeps the caveat — it recommends treating them as gaps but flags it as
  a judgement call, not a certainty. It points; it doesn't overrule.
- **We said out loud that the AI was down.** The easy, dishonest move would be to quietly
  generate a plausible-looking tip. We didn't. The record shows exactly what happened and
  that the real work — the plan — stands on its own.
- **Still can't peek at the final exam.** The Planner is actually *further* from the data
  than the surveyor — it never even opens the practice set, only reads the summary — and
  an automated check fails the build if it so much as mentions the locked test.

## Where this leaves us

Two of the crew's real specialists are now on the job, the plan for every dataset is
written and reproducible, and all **141** automated checks are green. The assembly line
now surveys the data *and* decides how to tackle it.

Tomorrow (Day 9) the **builders** arrive — the Feature Engineer and the Trainer. They take
this plan, write and safely run real code, train the models, and produce the crew's very
first honest scores.
