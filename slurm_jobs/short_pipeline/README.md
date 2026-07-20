# short_pipeline

Copies of the TRD-prediction pipeline wired for the **c3_short** partition (9-hour
wall) with **LLM judging OFF**. Built to get classical-ML + nearest-neighbor
prediction results out fast, while the slow neighbor-judging run finishes on
`c3_accel`.

## How it stays out of the live run's way

- **The real `.env` is never edited.** Each job sources `.env`, then sources
  `short_overrides.env` with `set -a` so those exports win in the shell. Every
  `load_dotenv()` in this repo uses `override=False`, so Python keeps the
  exported values.
- **All `ARTIFACTS_DIR`-derived outputs are redirected** to
  `${ANALYSIS_DIR}/artifacts_short` (narratives, feature parquet, embeddings.db,
  results). The live run's `${ANALYSIS_DIR}/artifacts` tree is untouched.
- **JSON build is skipped.** The sliced patient JSONs under `ANALYSIS_DIR` are
  model-independent and already exist; the embedding job reuses them read-only
  and regenerates the artifact side into the sandbox.
- Repo-side results mirror to `results/short_pipeline/...`, not the main
  `results/<encoder>/<judge>/...` path.

## Run it

    sbatch slurm_jobs/short_pipeline/trd_prediction_orchestrator_short.sbatch

Chain: embedding → prediction (array 0-0) → analysis. Cancel everything with:

    bash slurm_jobs/short_pipeline/cancel_short_pipeline.sh

## Testing a different encoder

Uncomment and edit `EMBEDDER_MODEL_NAME` / `EMBEDDER_MODEL_PATH` in
`short_overrides.env`; `EMBEDDINGS_DIR` and `RESULTS_DIR` follow automatically.

## Knobs

- `RUN_ABLATIONS` (default 0): the semantic-feature ablation re-embeds the whole
  cohort once per spec (5 specs) and will likely exceed the 9h wall. Turning it
  on also re-enables `ablation_runner` in the analysis job.
- Resource asks respect c3_short's `MaxMemPerCPU=12000 MB`. The prediction job
  uses 32 CPUs for 256G (the main pipeline's 2-CPU/256G ask would be rejected).
