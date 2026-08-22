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


def select_specs() -> list[dict]:
    """Decide which ablation specs this process should embed.

    Three modes, in precedence order. ABLATION_SPEC_IDS names specs explicitly as
    a comma-separated list of ids, which is what a backfill wants: appending one
    spec to the slate should not mean re-embedding the five already on disk, and
    naming the spec is safer than naming its array index. SLURM_ARRAY_TASK_ID
    selects the spec at that index, so one GPU handles one full-cohort pass at a
    time (see the short-pipeline ablation array). With neither set, every spec is
    embedded in series -- the original behavior, and what the full pipeline does.

    Returns:
        list[dict]: The ABLATIONS entries to embed, in slate order.

    Raises:
        ValueError: If ABLATION_SPEC_IDS names an id that is not on the slate.
        IndexError: If SLURM_ARRAY_TASK_ID falls outside the slate.
    """
    known = {spec["id"] for spec in ABLATIONS}

    requested = os.environ.get('ABLATION_SPEC_IDS', '').strip()
    if requested:
        wanted = {spec_id.strip() for spec_id in requested.split(',') if spec_id.strip()}
        unknown = sorted(wanted - known)
        if unknown:
            raise ValueError(
                f"ABLATION_SPEC_IDS names {unknown}, which are not on the ablation "
                f"slate. Known ids: {sorted(known)}"
            )
        # Slate order, not the order written in the variable, so the logs read
        # the same way however the variable was set.
        return [spec for spec in ABLATIONS if spec["id"] in wanted]

    task_id = os.environ.get('SLURM_ARRAY_TASK_ID')
    if task_id is not None:
        idx = int(task_id)
        if not 0 <= idx < len(ABLATIONS):
            raise IndexError(
                f"SLURM_ARRAY_TASK_ID={idx} out of range for {len(ABLATIONS)} ablation specs "
                f"(valid array is 0-{len(ABLATIONS) - 1})"
            )
        return [ABLATIONS[idx]]

    return list(ABLATIONS)


def main():
    baseline_narratives_dir = Path(os.environ['NARRATIVES_DIR'])
    baseline_embeddings_dir = Path(os.environ['EMBEDDINGS_DIR'])

    specs = select_specs()
    print(
        f"Embedding {len(specs)} of {len(ABLATIONS)} ablation spec(s): "
        f"{', '.join(spec['id'] for spec in specs)}",
        flush=True,
    )

    for spec in specs:
        embed_spec(spec, baseline_narratives_dir, baseline_embeddings_dir)


if __name__ == "__main__":
    main()
