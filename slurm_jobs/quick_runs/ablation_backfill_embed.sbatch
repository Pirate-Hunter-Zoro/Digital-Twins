#!/bin/bash
#SBATCH --job-name=ablation_backfill_embed
#SBATCH --partition=c3_short
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH -o slurm_jobs/quick_runs/logs/ablation_backfill_embed_out.txt
#SBATCH -e slurm_jobs/quick_runs/logs/ablation_backfill_embed_err.txt
#SBATCH --gres=gpu:1

# =============================================================================
# Stage 1 of 2: backfill the narratives and embeddings for ablation specs that
# were APPENDED to ablation_registry.ABLATIONS after the cohort had already been
# embedded.
#
# Adding a spec to the slate does not invalidate anything already on disk. The
# narrative builder skips any per-spec narrative that exists (SCRUB_NARRATIVES=0)
# and donor pairings are seeded per spec id, so the five specs already published
# are not touched, re-shuffled, or re-embedded. Only the named spec's narratives
# are written, and only its embeddings are forged -- one full-cohort pass, ~3h on
# one A40, which is why this is its own job rather than part of the analysis job.
#
# Which specs: set ABLATION_SPEC_IDS to a comma-separated list of ids before
# submitting, or leave it and the default below applies. Naming the spec rather
# than its array index is deliberate; the index is positional and easy to get
# wrong once the slate has grown.
#
#   sbatch --export=ALL,ABLATION_SPEC_IDS=permute_treatment_exposure \
#     slurm_jobs/quick_runs/ablation_backfill_embed.sbatch
#
# Then run ablation_backfill_score.sbatch with an afterok dependency on this job.
#
# c3_short enforces MaxMemPerCPU=12000 MB; 128G / 16 CPUs = 8 GB/CPU is legal.
# =============================================================================

# Exit immediately if command fails
set -e

# ---- Conda configuration ----
CONDA_ENV=/media/studies/ehr_study/analysis/mferguson/venvs/embedder_pipeline

# ---- Boilerplate env/bootstrap ----
module purge
module load Anaconda3/2025.06-0
eval "$(conda shell.bash hook)"

# Conda often dislikes 'set -u'—use this safe pattern:
set +u; conda activate "${CONDA_ENV}"; set -u || exit 10

# Load env variables
ENV_FILE="${SLURM_SUBMIT_DIR}/.env"
if [[ ! -f "${ENV_FILE}" ]]; then
  echo "ERROR: ${ENV_FILE} not found." >&2
  exit 11
fi
set -a; source "${ENV_FILE}"; set +a

# The .env does not define ABLATION_SPEC_IDS, so a value passed in through
# --export survives the sourcing above; this only supplies the default.
export ABLATION_SPEC_IDS="${ABLATION_SPEC_IDS:-permute_treatment_exposure}"
echo "Backfilling ablation spec(s): ${ABLATION_SPEC_IDS}"

# Refuse to run if the baseline embeddings are absent: the whole point of a
# backfill is that it adds to a completed run, and forging the new spec against a
# half-built baseline would produce deltas against nothing.
if [[ ! -f "${EMBEDDINGS_DIR}/embeddings.db" ]]; then
  echo "ERROR: ${EMBEDDINGS_DIR}/embeddings.db not found. Run the baseline embedding pipeline first." >&2
  exit 12
fi

# Writes the baseline narrative only if missing, then one directory per spec,
# again only where missing. With SCRUB_NARRATIVES=0 this is incremental.
python -m scripts.pipeline.narratives.forge_narratives

# Embeds only the spec(s) named above.
python -m scripts.pipeline.embeddings.forge_ablated_embeddings
