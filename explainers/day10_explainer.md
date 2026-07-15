# Day 10, in plain English — the reviewer joins, and the line becomes a team

## The one line

Until today the crew worked like an assembly line: survey the land, draw the plan,
prepare the materials, build. Today the **reviewer** showed up — the one who inspects
what the builder made, says what's actually wrong with it, and sends the strategist
back with a specific fix. That single hire turns a straight line into a team that
learns from its own first attempt.

## The new worker

**The Critic** is the seasoned reviewer who looks at the finished first draft and asks
the questions a good one always asks:

- Is this model *too jumpy* — does it behave very differently depending on which slice
  of the practice data you check it on? (overfitting / instability)
- Is it *too weak* — barely better than a coin flip? (underfitting)
- Is the score *suspiciously perfect* — the tell-tale sign the model is secretly
  peeking at the answer? (leakage)
- Did we handle the fact that one outcome is much rarer than the other? (imbalance)
- Are we even grading against the right yardstick? (wrong metric)

For each problem it finds, it doesn't just grumble — it writes the strategist a
**specific instruction** ("pull the complexity down and add more discipline",
"drop that leaky column", "force the rare-class handling on"). And crucially, the
strategist already knew how to act on exactly those instructions — we wired that side
up two days ago. So the moment the reviewer speaks, the plan actually changes.

## The interesting part: knowing when to *stop*

Here's the twist. When we ran the real reviewer on all five datasets, it looked at
every one and said the same thing: **"This is clean — ship it."** No changes, one pass,
done.

That sounds anticlimactic until you see what it replaced. The placeholder reviewer we'd
been running always said "do it again" no matter what — so every practice run wastefully
rebuilt the model three times before quitting. The real reviewer rebuilds it **once** when
there's nothing to fix. Knowing when *not* to send everyone back to the drawing board is
just as much the point as knowing when to. A reviewer who demands endless revisions on
already-good work isn't rigorous — they're just expensive.

It also quietly tells us something good: the strategist's plans are already sound enough
that there's no easy mistake left to catch. That's a compliment to the earlier crew, not
a gap in the reviewer.

## "But does the loop actually loop?"

Fair question — if every real dataset is clean, how do we know the send-it-back machinery
works? Because we tested it directly: we handed the reviewer a run with a deliberately
planted flaw and watched the whole crew respond for real — the strategist redrew the plan,
the builder rebuilt, and *then* the reviewer, seeing the second attempt was clean, called
it done. It stopped on its own, well before hitting the hard safety limit. So the feedback
loop genuinely opens, does its work, and closes itself — we just don't need it on data this
tidy.

## The honest fine print

- **"Too jumpy" is judged fairly, not by peeking.** The reviewer only ever looks at the
  practice data — never the sealed final exam. So when it flags instability, it means the
  model's scores wobble across different slices of practice, which is all it can honestly
  see yet. The real "does it hold up on the exam" check is still a few days away, and the
  reviewer is *forbidden* from touching that exam early.
- **The loop can't spin forever.** Two separate brakes: the reviewer stops itself when
  things are clean or when extra rounds stop helping, and a hard ceiling stops everything
  regardless. A crew stuck in an infinite "do it again" is impossible by construction.
- **The AI narrator was out again.** We ask the reviewer for a short written second opinion
  when an AI is available; the provider was locked out for the third day running. As
  designed, the crew simply noted it and leaned entirely on its own rock-solid checklist —
  no AI got a vote on whether to redo the work.

## Where this leaves us

Six of the crew's seven specialists are now real — survey, plan, prepare, build, **review** —
and the feedback loop between them is closed and tested (all **188** automated checks green).
The team can now take a dataset, build a model, honestly critique its own work, and decide
for itself whether to try again.

Tomorrow (Day 11) the last two roles arrive: the **Ensembler**, who combines the best
attempts into one stronger model, and the **Reporter**, who writes up the result and the
model's "nutrition label." That's the Phase-2 finale — the first time the whole crew, every
role real, runs a dataset from raw to finished model on its own.
