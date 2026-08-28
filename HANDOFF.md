# HANDOFF

Last updated 2026-08-28, 11:40 (post-meeting planned work appended).

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

## 2026-08-28 — the interval finally has a picture

The confidence interval existed only as JSON keys. Nothing plotted it, and the
one figure that *looked* like it plotted it — the pooled per-patient histogram's
orange span — is a spread over patients, wider than the real interval by roughly
the square root of the sample size. So the only available visual overstated the
spread of the estimate.

Three edits, both files under `scripts/pipeline/counterfactual/`:

- `bootstrap_effect` returns a **third** element — a dict of field name to an
  array of that field's value *one per surviving draw*, covering `ate_trimmed`,
  `ate_overlap_weighted` and both trimmed shares. The interval closure now reads
  from that dict instead of rebuilding arrays, so what gets plotted and what gets
  reported come from one object.
- New `plot_ate_sampling_distribution` in `core.py`: bars count **draws**, the
  2.5/97.5 span is shaded, the point estimate and zero are marked, and an
  on-axes callout says whether the band spans zero.
- `run_one.py` writes `bootstrap_draws.csv` (4 fields × 2 schemes) plus four
  figures per contrast — both estimands × both schemes.

**Ran directly, no sbatch.** `compute305` has 96 cores; with
`SLURM_CPUS_PER_TASK=24` all three contrasts took 4m20s, so queueing would have
been slower. `venvs/embedder_pipeline/bin/python` works without the
module/conda dance.

Verified: the CSV percentiles reproduce the JSON bounds to five decimals in all
six contrast × scheme cells, and `snri_vs_ssri/effect_results.json` still hashes
to `0567279a…` — identical to job 2066883, so the figures moved no number.

Two things a successor should know. Test-set resampling adds essentially nothing
to the width; model-estimation uncertainty is the whole budget, which means more
*training* patients would narrow these intervals and more test patients would
not. And `ate_overlap_weighted` still has **no** figure showing the population it
averages over — both per-patient histograms are fed `in_band_effects`, i.e. the
hard-trimmed population unweighted. A weighted counterpart is drawable but needs
the propensity column carried alongside the pooled contrasts.

`results/` is line 11 of `.gitignore`. Nothing copied there will ever appear in a
source-control pane; check the filesystem. And `/home/<user>/TRD-EHR` and
`/mnt/dell_storage/homefolders/<user>/TRD-EHR` are the same directory.

Agenda is unchanged: the `<<< RESUME HERE >>>` marker still points at the
falsification battery.

## 2026-08-28, post-meeting — three specified, none built

Documentation only. No code written for any of these; specs live in the
`TRD-EHR` README's counterfactual section and as `- [ ]` items in
`~/Research-Journey/planning/TRD-EHR_TODO.txt`.

**A misreading worth not repeating.** `reference_arm_n` and `comparison_arm_n`
in `effect_results.json` are *both test-side* — the two arms of the eligible test
population, summing to `n_eligible`. Neither is a training count. The training
population is recorded nowhere at all, which is the actual gap.

Measured while checking that: the split is 80/20 on 42,579, and arm shares plus
TRD rates are stable across the two sides to within about a point. The frozen
split is well balanced; nothing in the artifacts establishes that.

1. **Population report** — train and test sides separately: total rows, per-arm
   count and share, per-arm TRD events and rate, unmapped count. Plus the
   per-arm **retained** counts: `*_trimmed_count` and `*_trimmed_share` are the
   patients *removed*, nothing states the survivors, and mistaking one for the
   other misreads the analysis population by 3x or more (`bupropion_vs_ssri`
   rests on 3,721 SSRI patients, not 1,245). Report arm ratios before *and*
   after trimming — the floor cuts the larger arm harder, so the trim shifts
   composition as well as shrinking it: 4,966:934 in, 3,721:822 out.
2. **Propensity histogram coloured by arm**, with the band edges drawn. This
   also explains the trim asymmetry, which is structural: e(x) is oriented to
   the comparison arm, that arm is ~20% of the population, so the distribution
   is centred near 0.20 and a 0.10 floor cuts into its bulk while the 0.90
   ceiling barely touches the minority arm. Open decision on whether the
   symmetric band stays — build the figure before deciding.
3. **5-fold CV over the whole pipeline**, folds stratified on arm and TRD label
   jointly. ~22 minutes at current speeds, so this is a design decision and not
   a compute one. Two open questions recorded: whether CV replaces the shared
   frozen split or sits beside it, and how fold results combine (recommendation:
   fold-averaged point estimate, bootstrap interval, and between-fold spread
   reported as three separate quantities, never fused).

The `<<< RESUME HERE >>>` marker has **not** been moved — it still points at the
falsification battery. These three are additions, and the priority between them
and the battery has not been set.
