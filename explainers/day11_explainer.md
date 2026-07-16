# Day 11, in plain English — the last two hires, and the crew runs on its own

## The one line

The team was almost complete: surveyor, strategist, two builders, and a reviewer. Today
the **final two** showed up — the one who tries combining the best attempts into something
stronger, and the one who writes up the finished result in a clear, honest report. With
them, the crew can now take a raw dataset and hand you back a trained model *and its own
write-up*, start to finish, with nobody steering.

## The two new workers

**The Ensembler** is the coach who asks: *would our three best players, pooling their
judgement, beat our single strongest player alone?* Sometimes a committee outvotes its best
individual; sometimes it just drags the star down. So the coach doesn't guess — it runs the
committee and the star side by side on the same practice drills and **keeps whichever
actually scores higher**. Crucially, it keeps the committee *only* if it genuinely wins;
otherwise it sticks with the single best player. That one rule means combining can never
make things worse — the worst case is "we kept the player we already had".

**The Reporter** is the one who writes the honest write-up — the model's "nutrition label".
It doesn't do any thinking of its own (on purpose — a report that could make things up would
wreck the whole point). It just lays out, plainly: what the final model is and why it was
chosen, how each candidate scored, what the reviewer decided, and — in bold — the caveats a
reader must not miss.

## The interesting part: the coach said "no" every time — and that's the win

Here's what actually happened when we ran the full crew on all five datasets. The coach looked
at the committee-vs-star comparison and, on **every single one**, said: *"the committee is
worse — keep the star."* So the crew shipped its single best model each time.

That sounds like the new hire did nothing. It's the opposite. Our builders had already
**carefully tuned** each individual model, so the strongest player was genuinely excellent —
and blending it with two weaker teammates (an equal-weighted vote) just watered it down. A
coach who insists on a committee *no matter what* would have shipped a worse model five times
out of five. Ours refused, and the crew is provably never worse off for having asked.

And to be sure the "combine them" machinery actually works when it *should*, we checked the
other regime too: with the players **un-tuned** (weaker, and closer to each other in strength),
the committee genuinely wins — on the credit dataset it beats the single player by a small but
real margin, and the coach happily keeps it. So both answers are real: combine when it helps,
decline when it doesn't.

## What the crew produces now — and what the numbers honestly are

For the first time, the whole team ran a dataset from raw to finished model with **every role
real** — survey, plan, prepare, build, review, combine-or-not, write up. Out came a trained
model and a model card for each of the five datasets.

But we're keeping the same discipline we've kept all along: every score is still a
**practice-round score** (cross-checking on the training data), clearly labelled as such —
**not** the final-exam score. We have a sealed exam (the held-out data) that nobody on the crew
is allowed to touch, and the real test — *does this team actually beat a lone expert and an
off-the-shelf tool?* — happens tomorrow, on that exam. Today's milestone is that the crew is
**complete and runs itself**, honestly.

## The honest fine print

- **"We kept the single model" is a feature, not a shrug.** The job was a coach who decides
  *soundly* — takes the committee when it helps, refuses it when it doesn't. Both happen on real
  data. That the tuned runs didn't need a committee is a compliment to the builders.
- **Practice score, not exam score — still.** Every number today is cross-checking on the
  training data, labelled as such, and an automatic check confirms the sealed exam is **still
  untouched** after each run (all five: sealed and intact).
- **The report can't lie because it can't invent.** The write-up has no AI doing "creative"
  summarising — it only restates decisions the specialists already made, so there's nothing for
  it to hallucinate. And it puts the caveats (practice-not-exam, any missing AI second-opinions)
  in plain sight, not the footnotes.

## Phase 2 is complete

That's the end of Phase 2 — the "build a real multi-agent crew" chapter. Seven specialists, all
real, with a genuine review-and-retry loop between them, and **214** automated checks green. Over
the phase the crew learned to survey a dataset, plan the modelling, prepare features, build and
grade models, critique its own work and decide whether to try again, combine its best attempts
when that helps, and write up the result — all without loading the sealed exam even once.

Tomorrow (Day 12) opens Phase 3, the part this whole project has been building toward: we finally
**open the sealed exam** and put the crew head-to-head against a lone expert and an off-the-shelf
AutoML tool. Everything so far has been the crew getting ready. Now we find out if the team is
actually better than the alternatives — honestly, on data it has never seen.
