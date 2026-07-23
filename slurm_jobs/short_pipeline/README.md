# short_pipeline

Copies of the TRD-prediction pipeline wired for the **c3_short** partition with
**LLM judging OFF**. Built to get classical-ML + nearest-neighbor + ablation
prediction results out for a second encoder, while the slow neighbor-judging run
finishes on `c3_accel`.

## How it stays out of the live run's way

- **The real `.env` is never edited.** Each job sources `.env`, then sources
  `short_overrides.env` with `set -a` so those exports win in the shell. Every
  `load_dotenv()` in this repo uses `override=False`, so Python keeps the
  exported values.
- **Outputs share the real `artifacts/` tree, keyed by encoder name.** This is
  the main pipeline in substance -- same cohort, same `${ARTIFACTS_DIR}` -- just
  run on `c3_short`. Separation from the live 8B run comes from the different
  `EMBEDDER_MODEL_NAME` (`Qwen-Qwen3-Embedding-4B`): `embeddings.db` and results
  are keyed by encoder, while the model-independent narratives and feature
  parquet are shared (deterministic, identical content). There is no longer an
  `artifacts_short` sandbox.
- **JSON build is skipped.** The sliced patient JSONs under `ANALYSIS_DIR` are
  model-independent and already exist; the baseline job reuses them read-only and
  regenerates narratives / vectors / embeddings for the 4B encoder.
- Repo-side results mirror to the normal `results/<encoder>/<judge>/...` path
  (the 4B encoder name keeps it distinct from the live 8B mirror), done by the
  terminal `mirror` job once both analysis arms succeed.

## DAG

    baseline_embed ─┬─> prediction (array 0-0) ──> analysis ──┐
                    │                                          ├─> mirror
                    └─> ablation_embed (array 0-N%1) ──> classical_ml ──┘

The `analysis` (neighbor arm) and `classical_ml` arms are INDEPENDENT and run in
parallel; the single results rsync lives in the terminal `mirror` job so it only
copies a complete result set.

- **baseline_embed** (`run_embedding_pipeline_short.sbatch`, GPU): narratives +
  feature vectors + the **baseline** embed. `forge_narratives` also writes every
  per-spec **ablated narrative dir** (CPU), so the ablation array has narratives
  to read.
- **prediction** (`run_trd_prediction_pipeline_short.sbatch`, CPU, no vLLM):
  baseline neighborhoods. Needs only the baseline embed, so it runs in parallel
  with the ablation array.
- **ablation_embed** (`run_ablation_embedding_array_short.sbatch`, GPU): one
  spec per array task, submitted `--array=0-N%1` where N is derived from
  `ablation_registry.ABLATIONS`. The `%1` throttle means **only one GPU is ever
  held at a time** -- the specs embed in series, each a single ~3h full-cohort
  pass that clears the wall. No single job embeds the cohort more than once,
  which is what blew the wall in the old all-in-one embedding job. Raise `%1` to
  `%N` if more GPUs free up and you want them parallel.
- **analysis** (`run_trd_prediction_analysis_short.sbatch`, CPU, neighbor arm):
  `embedding_audit` + `analyze_trd_prediction`. Depends on **prediction only** —
  it never reads the per-spec embeddings, so it does not wait on the ablation
  array.
- **classical_ml** (`run_classical_ml_short.sbatch`, CPU, `c3` partition, 2-day
  wall): the `classical_ml` GridSearchCV sweep + `feature_importance` +
  `ablation_runner`. Split out of the analysis job (2026-07-22) because the
  EMBEDDED sweep alone (~2.5h for logistic regression on the 2560-dim embeddings)
  blew the shared 8h wall; then moved off `c3_short` to `c3` (2026-07-23) because
  gradient_boosting alone breached even the 9h c3_short ceiling, and `c3` is the
  only CPU partition with a longer (7-day) MaxTime. Independent of the neighbor
  arm, so it runs in parallel with
  prediction/analysis; depends on the baseline embed plus the whole ablation
  array (for `ablation_runner`). Fitted searchers are joblib-cached
  (`SCRUB_TRAINED_MODELS=0`), so a re-run resumes past any classifier that
  already finished.
- **mirror** (`run_results_mirror_short.sbatch`, CPU): the single results rsync,
  `afterok` on **both** analysis and classical_ml, so it copies a complete
  result set into `results/<encoder>/<judge>/` exactly once.

Each ablation task fails in isolation: a dead spec does not block the other
array tasks, only `classical_ml` (via its `afterok` on the array) and, in turn,
the terminal mirror.

## Run it

    sbatch slurm_jobs/short_pipeline/trd_prediction_orchestrator_short.sbatch

Cancel the whole chain (regenerated on each orchestrator run) with:

    bash slurm_jobs/short_pipeline/cancel_short_pipeline.sh

## Testing a different encoder

Edit `EMBEDDER_MODEL_NAME` / `EMBEDDER_MODEL_PATH` in `short_overrides.env`;
`EMBEDDINGS_DIR` and `RESULTS_DIR` follow automatically.

## Knobs

- `RUN_ABLATIONS` (currently 1): when on, the orchestrator launches the ablation
  embedding array and the analysis job runs `ablation_runner`. When off, both are
  skipped and analysis depends on prediction alone.
- Resource asks respect c3_short's `MaxMemPerCPU=12000 MB`. The prediction job
  uses 32 CPUs for 256G (the main pipeline's 2-CPU/256G ask would be rejected);
  the embedding jobs use 16 CPUs for 128G (8 GB/CPU).
