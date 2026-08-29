# HANDOFF

Written 2026-08-28, end of session.

**This repo is a project, not a course.** The agenda is the `<<< RESUME HERE >>>`
marker in `~/Research-Journey/planning/TRD-EHR_TODO.txt`, never the README's
architecture headings. `tutorboard.json` carries **`"stance": "do"`**: write the
code, run it, commit it. No override phrase is needed and none should be asked
for.

## Start here

```
squeue -u mferguson
tail slurm_jobs/review/logs/parity_*_out.txt
```

The **representation-parity re-run** is the only thing still in flight: `2067645`
(feature arm, c3_short) and `2067640_0/_1` (narrative control and parity arms,
c3_accel, ~4.5h each in their embed stage). When all three are done:

```
python -m scripts.pipeline.review.parity.summarize
```

then fill in Supplement S12.3 in the Research-Journey repo. Its decision rule was
fixed before the numbers existed and must not be renegotiated after seeing one.

## The review package

`scripts/pipeline/review/` holds four analyses, each with an sbatch in
`slurm_jobs/review/`. All of them run against the frozen published artifacts and
write under `ARTIFACTS_DIR/review/<name>/`, mirrored to `results/review/<name>/`
by their job. **None may write into `RESULTS_DIR`.** That is the safety property,
not a convention: these are re-analyses of published numbers, so a run that could
overwrite one would make the comparison unfalsifiable.

Three are complete: `holdout_representativeness`, `judge_prompt`, `subgroups`.

## Things that will bite you

**Two flags default to published behaviour and must stay that way.**
`NARRATIVE_INCLUDE_HISTORY_LENGTH` (off) changes what a narrative *says*, so
setting it invalidates every embedding, neighbour set and cached judgement.
`load_feature_matrix(include_vitals=)` and `make_classifier(impute_numeric=)`
(both False) reproduce the published pipeline exactly.

**`render_narrative` must keep its `sorted()` calls.** The comorbidity,
prescribing-safety and prior-trial dictionaries are built by iterating Python
sets, so before the sort the rendered order varied between processes and the
published narratives are not byte-reproducible. Treat every `sorted()` in that
function as load-bearing.

**Read the pipeline, not the manuscript's description of it.** Three of this
round's findings — the anchor definition, the vitals drop, the raw-versus-collapsed
sociodemographic fields — were cases where the paper described its own code
wrongly.

**c3 and c3_short share nodes but not scheduling policy.** c3_short has
`PriorityTier=20` against c3's 10, and tier is a hard ordering: every job in a
higher tier is considered before any job in a lower one. Two review jobs sat
pending on c3 with no projected start and began within three seconds of
`scontrol update JobId=<id> Partition=c3_short`. The review sbatch files default
to c3_short; the cost is a 9-hour wall.

## Watch this on the parity run

An untuned smoke check moved feature-arm logistic regression from 0.629 to 0.643
by adding the vitals alone, with no grid search — roughly twice the size of the
head-to-head effect under test. If the tuned number holds, the head-to-head
**null gets safer** and the +0.028 embedded logistic-regression gain is what comes
under threat. That is the opposite of what "the feature arm improved" sounds like.

The narrative arms were submitted before the sbatch stage order was fixed, so they
run the neighbour stage before the classifier fit. If one dies there, its
embeddings survive and stage 4 is standalone — the command is in the TODO.
