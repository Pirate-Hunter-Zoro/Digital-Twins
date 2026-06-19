"""Stage-1 coordinator.
"""
from scripts.embedder_investigation.narratives.generator import generate_deterministic_narratives

def run() -> None:
    print("Running deterministic narrative generation...", flush=True)
    generate_deterministic_narratives()
    print("[Narratives] complete.", flush=True)