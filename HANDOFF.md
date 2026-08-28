# HANDOFF

Last updated 2026-08-27, 20:35.

**This repo is a project, not a course.** The agenda is the `<<< RESUME HERE >>>`
marker in `~/Research-Journey/planning/TRD-EHR_TODO.txt`, never the README's
architecture headings.

## Where the work got to

Paper 2, `scripts/pipeline/counterfactual/`. The bootstrap now **refits all three
models inside every draw**. `estimate_effect` is gone, split into
`summarize_effect`, `pooled_training_frame`, `resample_populations`,
`estimate_once`, `bootstrap_effect`. Run twice per contrast — schemes
`estimation` and `total` — so CI keys carry a scheme suffix. `run_one.py` and
`slurm_jobs/pipeline/counterfactual/counterfactual_effects.sbatch` exist.

Written under the override phrase at 19:55 — **override expired; no-code rule is
back on.** Verified: imports clean under `embedder_pipeline`. Dry run held point
estimates (`+0.016151`) while `bupropion_vs_snri`'s band went ±0.003 → (−0.001,
+0.037), crossing zero.

**Blocked: the job has never been queued.** `sbatch` is not allowlisted headless.
Nothing has run; no post-refit numbers exist.

## Right, do not re-teach

The two schemes are nested not rival; bootstrap the whole pipeline and let the
trim population move; pooled across arms with the train/test split fixed;
`summarize_effect` moved faithfully (its only break — dropping both averages —
is fixed in the shipped code).

## Wrong

Nothing conceptual. The failure was procedural: three consecutive `[confused]`
turns asking whether the sbatch had been submitted, having decided to withhold
their explanation until it was. Told three times it was a permission wall.

## Next thing to teach

The question standing on card `0008`, unanswered: **what is
`bootstrap_effect_histogram_total.png` a picture of, and why is it wider than
`effect_histogram.png`?** It needs no numbers, so it breaks the deadlock. Wrong
answer to watch for: "the same histogram with more data."

## How they work

They read the top of a card and stop, so put the answer in line one. They ask for
the whole plan, then say it was too much — default to one step. Never run git
here. Documentation debt: three edits owed to `TRD-EHR_TODO.txt` (tick the refit
item at 640, tick `run_one.py` at 673, move the marker onto the falsification
battery) — the sandbox is scoped to `TRD-EHR` and blocks that path; do not route
around it.
