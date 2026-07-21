import os
from pathlib import Path
import subprocess

from dotenv import load_dotenv
load_dotenv()

from scripts.data_loading.ablation_registry import ABLATIONS


def embed_spec(spec: dict, baseline_narratives_dir: Path, baseline_embeddings_dir: Path) -> None:
    """Re-embed the cohort for a single ablation spec.

    Points NARRATIVES_DIR / EMBEDDINGS_DIR at the spec's subdir and runs
    forge_embeddings in a fresh subprocess (PatientEmbedder snapshots
    EMBEDDINGS_DIR at construction, so a new interpreter per spec is required).
    """
    spec_id = spec["id"]
    ablation_narrative_dir = baseline_narratives_dir / spec_id
    os.environ['NARRATIVES_DIR'] = str(ablation_narrative_dir)
    ablation_embeddings_dir = baseline_embeddings_dir / spec_id
    os.environ['EMBEDDINGS_DIR'] = str(ablation_embeddings_dir)
    os.makedirs(ablation_embeddings_dir, exist_ok=True)

    print(f"Ablating on {spec}...", flush=True)
    print(f"Narratives: {os.environ['NARRATIVES_DIR']}", flush=True)
    print(f"Embeddings: {os.environ['EMBEDDINGS_DIR']}", flush=True)

    # With all the proper .env changes made, forge the embeddings with the given ablation
    subprocess.run(
        ["python", '-m', 'scripts.pipeline.embeddings.forge_embeddings'],
        check=True,
    )


def main():
    baseline_narratives_dir = Path(os.environ['NARRATIVES_DIR'])
    baseline_embeddings_dir = Path(os.environ['EMBEDDINGS_DIR'])

    # When launched as a SLURM array task, embed ONLY the spec at this index so a
    # single GPU handles one full-cohort pass at a time (see the short-pipeline
    # ablation array). Absent an array context, fall back to embedding every spec
    # in series (the original single-process behavior).
    task_id = os.environ.get('SLURM_ARRAY_TASK_ID')
    if task_id is not None:
        idx = int(task_id)
        if not 0 <= idx < len(ABLATIONS):
            raise IndexError(
                f"SLURM_ARRAY_TASK_ID={idx} out of range for {len(ABLATIONS)} ablation specs "
                f"(valid array is 0-{len(ABLATIONS) - 1})"
            )
        specs = [ABLATIONS[idx]]
    else:
        specs = ABLATIONS

    for spec in specs:
        embed_spec(spec, baseline_narratives_dir, baseline_embeddings_dir)


if __name__ == "__main__":
    main()
