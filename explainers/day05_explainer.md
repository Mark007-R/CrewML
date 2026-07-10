# Day 5, in plain English — building the assembly line before the workers

## The one line

Phase 1 built the scoreboard and the judges. Today Phase 2 begins, and the first
thing we built isn't a smart agent — it's the **assembly line** the agents will
stand on: the stations, the conveyor belt between them, and one crucial safety
switch that stops the line from running forever.

## Why build an empty assembly line first

The crew is seven specialists working in sequence: a **Profiler** who inspects the
data, a **Planner** who decides the approach, a **Feature Engineer** and a
**Trainer** who build and train the model, a **Critic** who checks the work and
sends it back if it's not good enough, and finally an **Ensembler** and a
**Reporter** who combine the best results and write them up.

Before hiring any of them, you install the conveyor belt and mark out where each
worker will stand. That's what today is. Every station currently has a cardboard
cut-out instead of a real worker — a "stub" that just says *"a real Profiler will
stand here later"* and passes the work along. That sounds trivial, but it's the
part that's easy to get subtly wrong, so it's worth getting right and testing
while it's still simple.

## The one part that's genuinely real: the feedback loop

Here's the interesting bit. The Critic isn't the end of the line — if it thinks
the model can be improved, it sends the work **back** to the Planner for another
round. That loop is the whole reason a *crew* might beat a lone AI: it gets more
than one attempt.

But a loop that sends work backwards has an obvious danger — what if it never
decides "good enough"? It would circle forever. So the real, working piece we
shipped today is the **safety switch**: the crew gets a fixed budget of retries
(three, by default), and once that's spent, the line moves on to finishing **no
matter what the Critic wants**. We deliberately built the cardboard Critic to be
never satisfied — it always demands another round — precisely so we could prove
the safety switch is what stops it, not luck.

Run the empty line and you can watch it happen:

```
profiler → planner → engineer → trainer → critic
         → planner → engineer → trainer → critic
         → planner → engineer → trainer → critic
         → ensembler → reporter → done
```

Three loops, then the switch trips and it finishes. Exactly right.

## Keeping it honest, structurally

Two things worth calling out, both about not fooling ourselves:

- **No fake numbers.** The cardboard Trainer refuses to report a score at all —
  it hands back a blank where the model's grade would go. That way no placeholder
  can ever be mistaken for a real result once we plug the real workers in.
- **The locked test set is now untouchable by design.** The crew is only ever
  told the *name* of its dataset, never handed the sealed final-exam set. We even
  added a test that scans the crew's own code and **fails the build** if it so
  much as mentions the locked test set. The no-peeking rule isn't a promise
  anymore — it's wired into the structure.

## Where this leaves us

The assembly line runs end-to-end, the feedback loop works, the safety switch
works, and 88 automated checks are green. We still haven't built a single real
agent — and that's the plan. You lay the track before you run the train.

Tomorrow (Day 6) we build the single most important tool the whole crew depends
on: a **safe sandbox** where the agents' code actually runs — with a time limit,
its output captured, and no way to escape and touch anything it shouldn't. Every
real worker we hire after that will reach for this one tool.
