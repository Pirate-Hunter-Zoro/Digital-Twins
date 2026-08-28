# HANDOFF

Last updated: 2026-08-27.

## This repository is a project, not a course

There are no chapters, no sections and no exercises here. **Do not build an
agenda out of the README's numbered Project Structure headings** — those describe
the pipeline's architecture, not a teaching order. An earlier sitting today made
exactly that mistake and opened a fictitious "Chapter 1 — Data Loading"; that
session is archived under `live/archive/20260827-151307-lesson` and should not be
resumed.

**Where the agenda actually comes from:** `README.md` names
`~/Research-Journey/planning/TRD-EHR_TODO.txt` as the live task list. That file
is authoritative and it marks the current item explicitly with a
`<<< RESUME HERE >>>` line. Read it, follow the marker, do not choose your own
next step. `~/Research-Journey` is a separate private repo sitting as a sibling
of this one in the home folder.

## Where the work is

**Paper 2, the counterfactual treatment-selection package**,
`scripts/pipeline/counterfactual/`. Design locked 2026-08-13. `core.py` currently
holds eligibility, the two-arm T-learner fit and scoring, gradeable metrics,
propensity and trimming, `estimate_effect`, and the effect histogram. First
numbers landed 2026-08-24 from a throwaway scratch script and are mirrored to
`results/counterfactual_pipeline/{key}/`. They are pre-falsification and not
reportable.

Recorded point estimates, for checking that a refactor did not move them:

| contrast | eligible | ate_trimmed (95% CI) | ate_weighted |
|---|---|---|---|
| snri_vs_ssri | 6,242 | +0.0337 (0.0320, 0.0352) | +0.0343 |
| bupropion_vs_ssri | 5,900 | +0.0407 (0.0389, 0.0427) | +0.0426 |
| bupropion_vs_snri | 2,210 | +0.0162 (0.0128, 0.0198) | +0.0134 |

## The change currently on the board

Card `0001` asks for the `RESUME HERE 2026-08-24` item: **refit the models inside
the bootstrap**, in `estimate_effect`, `scripts/pipeline/counterfactual/core.py`
lines 271–290.

The defect: the reference-arm and comparison-arm risk models (`core.py:95-98`)
and the propensity model (`core.py:215-216`) are all fitted once, before
`estimate_effect` is called. The draw loop therefore only resamples a frozen
`per_patient_effect` vector, so the interval captures the sampling variability of
a mean and omits model-estimation uncertainty entirely — which is the dominant
term, since both risk columns are predictions from fitted logistic regressions.
Hence bands of ±0.002 against per-patient effects spreading ±0.10.

What the card specified: all three models refit per draw; `estimate_effect` needs
access to `EligiblePopulations` to do it, and designing that route is the
student's structural decision; the shared-draw property must survive (both
averages from the same refit, or hard-vs-soft agreement stops meaning anything);
a degenerate single-class resample must land as `np.nan` alongside the existing
empty-band and zero-weight guards rather than crashing, with `bupropion_vs_snri`
the contrast at risk. Point estimates stay on the full-data fit; only the interval
changes. Seed stays as at `core.py:255`.

Success signals given to them: bands widen substantially, point estimates hold at
the table above, the hard-vs-soft gap stays within ~0.003, and the run stops being
instant (~1000 × 3 fits per contrast).

## What they were asked, and what has not happened yet

Card `0001` asked the question the TODO flags as *settle before coding*: **inside
each draw, do the test rows resample too, or only the training rows?**

They answered **"Can we do both?"** — which is right, and card `0002` says so and
works out what it costs. Do not re-teach this. What they saw, and what they should
be credited with: the two schemes are nested rather than rival. Scheme **A**
(estimation-only) resamples training and averages over the full fixed test frame;
scheme **B** (total) resamples training *and* test rows. B is the loop that exists
today with the refit added; A is that same loop with the row resampling taken back
out. One flag through the draw separates them.

Card `0002` also pinned three consequences: the shared-draw constraint scopes
*within* a scheme and must not be forced *across* them; the return grows from four
CI keys to eight (`core.py:298-304`), so the naming wants deciding before
`run_one.py` persists it; and in both schemes refitting the propensity model moves
`in_prob_interval`, so the `ate_trimmed` band is over a slightly shifting
population — inherent to bootstrapping a trimmed estimator, owed a sentence in the
write-up, not something to engineer around.

**Decided by them, 15:33:** bootstrap the *whole pipeline* — fit, score,
propensity, trim, average — and let the trimmed population move per draw. They got
there on their own reasoning ("standard process to bootstrap the whole pipeline"),
and they are right: the trimming rule is part of the estimator, so its variability
belongs inside the band. Do not re-argue this, and do not treat the moving trim
population as a defect to engineer around; card `0002` had raised it as a caveat
and this supersedes that framing.

**Still owed, deferred deliberately:** which scheme is the headline and which the
sensitivity (card `0002`). It is a reporting decision, not a construction one, so
the numbers can come first. Do not lose it — the paper must lead with one.

## Card `0003`: the refactor, step 1 of several

The obstruction: the estimator and the interval are one function. `estimate_effect`
computes the two point averages at `core.py:252-253` and then loops over draws of a
frozen vector at `core.py:277-287`. You cannot wrap a bootstrap around that, because
the loop is already inside it.

Step 1 asked of them — a pure extraction, no behaviour change:

- **The estimator.** A new function taking a post-`attach_propensity` risk frame and
  returning the trim report plus the two point averages. That is `core.py:232-253`
  plus the trim-report dictionary at `core.py:257-269`, lifted out whole. No
  generator, no draws, no CI keys. Suggested name `summarize_effect`; naming is
  theirs.
- **The harness.** What remains of `estimate_effect` keeps `N_BOOTSTRAP`, the
  `SEED`-seeded generator at `core.py:255`, `n_rows` at `core.py:254`, the two
  accumulators, the `np.nanpercentile` reduction at `core.py:289-290`, and the
  return assembly.

Self-check given: every point estimate, eligible count and trim share must come back
identical to the table above, **and the intervals must be unchanged too**, since
nothing the draws resample has been touched yet. Movement means something was
rewritten rather than moved.

**Answered 15:43: POOLED.** Draws come from the eligible population as one pool and
arm sizes wobble draw to draw. Settled; do not re-open. Note the boundary that
decision does *not* cross: the train/test split stays fixed, because
`create_train_test_split` / `test_patient_ids.txt` is shared project-wide with the
classical-ML and neighbour pipelines. Pooled means pooled **across arms**, within
train and within test separately.

They then asked for the whole plan up front, which the contract allows, so card
`0004` gives all five steps rather than one.

## Card `0004`: the full refactor, five steps

No new imports anywhere; everything needed is already at the top of `core.py`.

1. **`summarize_effect`** — as card `0003`. Move `core.py:232-253` + the trim-report
   dict at `core.py:257-269`. Check: all numbers identical, intervals included.
2. **`resample_populations`** — new. Population + generator + a resample-test
   boolean → a new population. Pool training by concatenating ref then comp (the
   same construction already at `core.py:213-214`, which should become a shared
   helper — the order defines the arm flag), draw with the generator's `integers`
   method, slice with **`.iloc`**, re-split by the resampled flag as at
   `core.py:77-80`. Test side: slice all three of matrix, labels and
   `test_comparison_flag` by one index set, or pass all three through untouched.
3. **`estimate_once`** — new, three calls: `score_counterfactual_risks` →
   `attach_propensity` → `summarize_effect`. The moving trim population falls out
   of this for free, since the propensity refit recomputes `in_prob_interval`.
4. **`bootstrap_effect`** — new, mostly the old loop. Keeps `N_BOOTSTRAP`, the
   `SEED` generator, the accumulators and `np.nanpercentile`. The pre-drawn index
   matrix at `core.py:274` goes away. Catch a failed pass, leave `np.nan`, and
   **count and return the failures**. Returns only the four CI keys.
5. **Rewire the caller.** Point estimates come from `summarize_effect` on the frame
   the caller already holds; `bootstrap_effect` takes the population. Call it twice
   — test resampling off (scheme A) and on (scheme B) — for eight CI keys. Name them
   with a scheme suffix before writing, since `run_one.py` persists those names.
   `estimate_effect` retires or becomes a thin merge.

**Verified by me, not by them (2026-08-27).** I ran a synthetic bootstrap-resampled
frame through a dtype-routed pipeline of the shape the README documents for
`make_classifier`. Results, all confirming the card: an 81-of-200 duplicated index
fits and predicts without complaint; `.set_index` onto a duplicated index works and
`.to_numpy()` on the contrast keeps the right length; boolean-mask arm splitting
works on a duplicated index; a single-class fit raises `ValueError`, which is what
step 4's guard must catch; and **`.loc` with one repeated label returned 3 rows**,
which is the combinatorial expansion step 2 warns about. Caveat: the project's own
`make_classifier` would not import under `causal_forest_env`
(`ModuleNotFoundError`), so a stand-in with the same dtype routing was used — the
pandas and sklearn semantics are confirmed, the specific project function is not.

**Deferred, non-blocking:** the trim shares move per draw too, so what single number
gets reported for `reference_trimmed_share` / `comparison_trimmed_share` — the
full-data value alone, or that value with its own percentile interval? The lopsided
~2× reference-arm trimming in both SSRI-referenced contrasts is one of the more
informative things in the output and is about to acquire error bars either way.
Decide before step 5.

## Step 1 in progress — card `0005`

**Pacing correction from them, 15:53, tagged `[confused]`: "one step at a time."**
Card `0004` gave all five steps because they explicitly asked for the whole plan,
and it was too much. Treat `0004` as the map only. Do not deliver more than one step
per turn again in this thread unless they ask for the plan in those words.

They have written `summarize_effect` at `core.py:222-258`. **The move is faithful** —
`core.py:225-239` and the trim report at `core.py:245-257` are the original lines,
arithmetic untouched, and they correctly left the row count and the `SEED` generator
behind in `estimate_effect` for the harness. Do not re-teach any of that.

**The one break, located and sent back unrepaired:** `core.py:242-243` bind
`ate_trimmed` and `ate_weighted`, and `core.py:258` returns `trim_report` alone, so
both floats fall out of scope. The function estimates the treatment effect and then
reports only how many patients it trimmed. Everything above the return is right.

Told them what has to come out, derived from the two consumers rather than dictated:
the caller needs the whole dict (it becomes the persisted artifact), and
`bootstrap_effect` needs only the two averages per draw. Also flagged the one silent
trap — their local is `ate_weighted` but the artifact key already on disk in
`results/counterfactual_pipeline/{key}/effect_results.json` is
**`ate_overlap_weighted`**; the strings differ and nothing catches a wrong one.
Noted the missing docstring and return annotation in passing.

**Question on the board:** which keys `bootstrap_effect` reads from that dict and
which it ignores — asked to confirm the shape landed rather than the keystrokes, and
because it is the reason step 4 is short. No answer yet.

Behind this item, in order: `run_one.py` plus its sbatch (owns array-index →
contrast pick, mirrors `causal/run_one.py`; `core.py` stays a library with no
main), then the four-check falsification battery. Both are blocked until the
interval is right, because they would only persist the wrong one.

## Notes on running this board

- The student writes all the code. Cards explain and ask; never put an
  implementation on the board. `tutorboard.json` sets `"mode": "code"`, and
  `AI_INSTRUCTIONS.md` §3 is not relaxed by the board.
- `board` is **not** in this session's Bash allowlist and a headless run cannot
  get approval, so `board recap`, `board next`, `board inbox` and `board finish`
  all fail. Write cards straight into `live/cards/` — the server serves the file
  the instant it exists. Naming is a four-digit zero-padded index, a hyphen, a
  slug, `.md`.
- The board is already running on compute301 port 8812; do not start another. The
  iPad address is the tailnet one in `live/BRIEF.md`.
- Standing rule from the user for this repository: **never run a git command
  here.** Commits go through the board's save-and-push button, and they are the
  user's — no assistant attribution anywhere in them.
