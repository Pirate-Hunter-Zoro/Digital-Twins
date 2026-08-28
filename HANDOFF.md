# HANDOFF

Last updated 2026-08-28, evening (Paper 1 review-round analyses built and running).

**This repo is a project, not a course.** The agenda is the `<<< RESUME HERE >>>`
marker in `~/Research-Journey/planning/TRD-EHR_TODO.txt`, never the README's
architecture headings. This repository's `tutorboard.json` carries
**`"stance": "do"`**: write the code, run it, commit it. No override phrase is
needed and none should be asked for.

## Where the work got to

Paper 1's review round of 2026-08-28 moved from prose into analysis. A new
`scripts/pipeline/review/` package holds four analyses, all of which run against
the frozen published artifacts and write under `ARTIFACTS_DIR/review/<name>/`,
never into `RESULTS_DIR`. That separation is the whole safety property: these are
re-analyses of published numbers, so a run that could overwrite one would make
the comparison unfalsifiable.

Landed and reported:

- `holdout_representativeness.py` — the held-out half against the training half.
  Max |SMD| 0.036 over 100 predictor rows, none reaching 0.1.
- `subgroups/` — subgroup discrimination and calibration, refitting nothing.
  **This one did not return the expected null**; see below.
- `plot_time_zero_timeline.py` — the study timeline, now manuscript Figure 1.

Running as of hand-off (`squeue -u $USER`):

- **2067636** `judge_prompt_cmp` — 5,000 cached pairs re-judged under the
  phenotype-free rubric. Slow because it shares compute306 with two embedders.
- **2067639** `parity_feature` — feature arm with vitals, queued on c3.
- **2067640/1** `parity_narr` — the two narrative arms, ~4.5h each in the embed
  stage.

Both in-flight runs have a `[PENDING]` marker waiting on them in the supplement
(S1.6 for the judge, S12.3 for parity), and both have their decision rule
written down **before** the numbers exist. Do not renegotiate either rule after
seeing a number. Finishing steps are itemized in the TODO.

## What the closed items found

**Subgroup performance is not a clean null, and the scoped-down version would
have missed it.** The plan was logistic regression on four groups, expecting no
significant difference. Sex delivered that. Race did not: all eight
White-minus-non-White AUC differences are positive, three of eight intervals
exclude zero (random forest at both representations, XGBoost on the feature
vector, each near +0.040), and the non-White calibration slope drops to 0.79 on
the feature vector against 0.98 for White patients, while the embedded
representation holds at 0.95/0.97. Logistic regression alone shows nothing at
either representation. Running all four classifiers cost nothing — their
predictions were already on disk — and it is the only reason the finding exists.

**The published narratives are not byte-reproducible, and that is our defect.**
`render_narrative` emitted its comorbidity, prescribing-safety and prior-trial
flag lists by iterating Python **sets**, whose order varies between processes. A
re-render on identical input produced identically informative but differently
ordered text. No published number is affected — each narrative was rendered once
and every embedding, neighbour set and judgement came from the text actually
written — but a re-render cannot be a byte-level control, which is why the parity
comparison carries a control arm. The renderer sorts now. Treat any `sorted()`
in that function as load-bearing.

**Read the pipeline, not the manuscript's description of it.** The field-level
crosswalk found two asymmetries nobody had listed: four sociodemographic fields
are collapsed to coarse categories in the feature matrix while the narrative
prints the raw recorded value, and categorical missingness is not dropped on the
feature side — an absent value becomes its own one-hot level. Both came from
reading `feature_vector.py` and `deterministic_narrative.py` against each other.
Two of this round's earlier findings had the same shape.

## Watch this on the parity run

An untuned smoke check moved feature-arm logistic regression from 0.629 to 0.643
by adding the vitals alone, with no grid search — roughly twice the size of the
head-to-head effect under test, pointing the way the crosswalk predicts. If the
tuned number holds, the head-to-head **null gets safer** and the +0.028 embedded
logistic-regression gain is what comes under threat. That is the opposite of the
reading a casual glance at "the feature arm improved" would give.

## Two flags default to the published behaviour and must stay that way

`NARRATIVE_INCLUDE_HISTORY_LENGTH` (off) changes what a narrative *says*, so
setting it invalidates every embedding, neighbour set and cached judgement.
`load_feature_matrix(include_vitals=)` and `make_classifier(impute_numeric=)`
(both False) reproduce the published pipeline exactly. The review package flips
them and writes elsewhere; nothing else should flip them.
