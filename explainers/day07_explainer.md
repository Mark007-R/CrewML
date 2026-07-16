# Day 7, in plain English — the first real worker walks in

## The one line

Yesterday we finished the workshop and its one power tool. Today we hired the
**first real specialist**: the one who opens the box of data and writes down
exactly what's inside — what the columns are, what's missing, how lopsided the
thing we're predicting is, and whether anything looks *too good to be true* —
before anyone tries to build a model.

## Why start with this worker

You would never let a builder start construction without first surveying the plot.
The **Profiler** is that surveyor. Everyone downstream — the Planner who picks the
strategy, the Trainer who builds the model — works off the Profiler's notes. If the
survey is wrong, everything built on it is wrong. So the rule for this worker is
strict: **measure, don't guess.**

## What the Profiler writes down

For each dataset it produces a tidy fact sheet:

- **The shape of things** — how many rows and columns, which columns are numbers vs.
  categories, and for each one: how much is missing, how many distinct values, the
  typical range.
- **The thing we're predicting** — for a yes/no or multi-way label, how many of each
  (is it lopsided?); for a number, its spread. On the credit dataset it noted the
  classes run about **2.3 to 1** — worth knowing, because a lazy model can look good
  just by always guessing the common answer.
- **Hidden missing values** — the sneaky one. Some datasets record "we don't know"
  as a **zero** instead of a blank. On the diabetes data the Profiler flagged that
  the insulin reading is **zero nearly half the time** — almost certainly "not
  measured," not a real zero. Catching this is the whole reason this dataset is in
  our test suite.
- **"Too good to be true" checks** — a quick sweep for columns that basically *give
  away the answer* (a leak), columns that are secretly just an ID number, exact
  duplicate columns, or duplicate rows. These are the mistakes that make a model look
  brilliant in practice and fail in the real world.

## The part we're proud of: it doesn't cry wolf

A cheap "leak detector" that flags everything is useless — people learn to ignore
it. So we tuned ours to be **quiet unless it's sure.** On the cleanest dataset
(kin8nm) it raised **zero** alarms — correctly. Then, to prove it isn't just asleep,
our tests hand it a rigged dataset with a planted "cheat" column, and it catches it
every time. Silent on clean, loud on real problems — that's the behaviour we want.

## A smart assistant, kept on a short leash

Here's the interesting bit. The Profiler has an **AI assistant** that reads the fact
sheet and writes a short, plain-English briefing for the next worker ("watch the
imbalance, double-check those zero-filled columns, you're being graded on X"). We
tried it live and it gave genuinely useful advice.

But we were deliberate about the division of labour: **the AI never touches the
numbers.** Every fact is computed by hand with ordinary math; the AI only
*interprets* what's already been measured. That way it can't make something up. And
if the AI is unavailable — no internet, or we switch it off — the Profiler shrugs and
carries on with its facts intact. The assistant is a bonus, never a crutch.

## The honest fine print

- **"Suspected," not "confirmed."** The hidden-missing flag is a heuristic. Some
  zeros are real (a patient really can have zero prior pregnancies). So the Profiler
  says *suspected* and spells out the caveat, leaving the final call to the next
  worker. It points; it doesn't overrule.
- **A small course-correction.** Yesterday's note guessed this worker would use the
  power tool from Day 6. It turned out not to need it — that tool is for running
  *risky, freshly-written* code, and the Profiler's work is plain, trusted
  bookkeeping. The power tool's real first customer arrives on Day 9. We're saying so
  out loud rather than pretending the original plan was right.
- **Still can't peek at the final exam.** Like everything in this project, the
  Profiler only ever opens the *practice* data, never the locked test set — and an
  automated check fails the build if it so much as mentions it.

## Where this leaves us

The first real crew member is on the job, all **120** automated checks are green,
and the assembly line now does genuine work at its first station instead of just
passing an empty box along.

Tomorrow (Day 8) we hire the **Planner** — the strategist who reads the Profiler's
survey and decides *how* to attack the problem: what to clean, which kinds of models
to try, and how to test them fairly.
