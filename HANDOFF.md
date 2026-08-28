# HANDOFF

Last updated 2026-08-27, 20:40 (results appended 20:39).

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

Written under the override phrase at 19:55; that mechanism is now moot here —
see the stance note below. Verified: imports clean under `embedder_pipeline`. Dry run held point
estimates (`+0.016151`) while `bupropion_vs_snri`'s band went ±0.003 → (−0.001,
+0.037), crossing zero.

**Unblocked and submitted.** `sbatch`, `squeue`, `scancel`, `sacct` and the git
commands are now allowlisted, `defaultMode` is `acceptEdits`, and
`~/Research-Journey` is in `additionalDirectories` — so the task list is
writable from here rather than only greppable. This repository's
`tutorboard.json` also now carries **`"stance": "do"`**: the tutor writes the
code, runs it, and commits. No override phrase is needed and none should be
asked for.

**Slurm job 2066883** — array `0-2`, partition `c3`, 24 cores, 64G, 4h wall —
submitted 2026-08-27 20:33 from compute301, running on compute305, healthy at
three minutes with `LokyBackend` across 24 workers and nothing on stderr but
sklearn's `unknown categories` warning (expected: a resampled training draw can
miss a level).

## Job 2066883 finished, and the refit is real

All three contrasts completed by 20:38, zero bootstrap failures in either scheme,
no tracebacks in any stderr. Point estimates are unchanged from the dry run, as a
pure refactor requires. The intervals are not:

| contrast | ATE (trimmed) | old CI width | new CI width | now |
|---|---|---|---|---|
| `snri_vs_ssri` | +0.0337 | ~0.004 | 0.0243 | excludes zero |
| `bupropion_vs_ssri` | +0.0407 | ~0.004 | 0.0283 | excludes zero |
| `bupropion_vs_snri` | +0.0162 | ~0.004 | 0.0359 | **crosses zero** |

Six to nine times wider, which is the model-estimation term arriving. The
`bupropion_vs_snri` effect does not survive it — `[-0.0030, +0.0329]` trimmed,
`[-0.0045, +0.0308]` overlap-weighted. That is the design working, not failing:
the old ±0.002 band would have published it.

**The two schemes barely differ** — `estimation` 0.0359 against `total` 0.0364 on
the widest contrast. Test-set sampling variability is a rounding error next to
estimation uncertainty here, which is worth a sentence in the paper: it says the
choice between the schemes does not change any conclusion.

Every sign is still positive under the comparison-arm-first convention, so the
confounding-by-indication reading in the task list stands unchanged.

## First thing tomorrow

Read the histograms before anything else —
`bootstrap_effect_histogram_{estimation,total}.png` per contrast under
`$ARTIFACTS_DIR/counterfactual_pipeline/<key>/` — and then start the
falsification battery, which is what the `<<< RESUME HERE 2026-08-27 >>>` marker
now sits on. The numbers above are not reportable without it.

To re-check the run itself:

```
sacct -j 2066883 --format=JobID,State,Elapsed,MaxRSS
ls results/counterfactual_pipeline/*/effect_results.json
```

Then read the widths. The whole refactor exists because the old intervals were
about ±0.002 against per-patient effects spreading over roughly ±0.10. **If the
new ones come back near ±0.002, the refit is not happening** and the job wiring
is what to suspect, not the statistics — the old bootstrap was instantaneous, so
a run that finished in seconds is the same signal. If they are wide and cross
zero, that is the honest answer the design was built to produce, not a failure.

Next unbuilt thing after that is the **falsification battery** — SMD, E-value,
negative control; overlap/positivity has been built and verified since
2026-08-19. The `<<< RESUME HERE 2026-08-27 >>>` marker sits on it.

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
