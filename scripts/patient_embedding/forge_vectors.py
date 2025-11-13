"""Stage 2 entrypoint.
Loads Stage-1 narratives, extracts sections, embeds, and writes vectors.
CLI-only; delegates to stage2.runner.run()."""

#!/usr/bin/env python3
# scripts/patient_embedding/forge_vectors.py
from __future__ import annotations
import sys
from pathlib import Path

# Preserve original sys.path behavior
sys.path.append("scripts")
sys.path.append(str(Path(__file__).resolve().parents[1] / "common" / "models"))

from patient_embedding.stage2.runner import run  # type: ignore

def main() -> None:
    run()

if __name__ == "__main__":
    main()