"""Stage-2 coordinator.
Initializes embedder, gathers narratives, runs embedding loop."""

from scripts.digital_twins.stage2.forge_vectors import forge

def run() -> None:
    forge()
    print("Stage 2 complete.", flush=True)