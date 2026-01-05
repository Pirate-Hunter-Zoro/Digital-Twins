"""Stage-1 coordinator.
"""
from scripts.digital_twins.stage1.generator import generate_deterministic_narratives

def run() -> None:
    print("Running deterministic narrative generation...", flush=True)
    generate_deterministic_narratives()
    print("[Stage1] complete.", flush=True)