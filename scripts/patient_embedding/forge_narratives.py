"""Stage 1 entrypoint.
Builds cohort + generates patient narratives via llama.cpp and writes JSON/MD under stage1_narratives/.
CLI-only; delegates to stage1.runner.run() without changing behavior."""

#!/usr/bin/env python3
# scripts/patient_embedding/forge_narratives.py
from __future__ import annotations
import sys
from pathlib import Path

# --- keep original sys.path behavior intact ---
sys.path.append("scripts")
sys.path.append(str(Path(__file__).resolve().parents[1] / "common" / "data_loading"))
sys.path.append(str(Path(__file__).resolve().parents[1] / "common" / "models"))

from patient_embedding.stage1.runner import run  # type: ignore

def main() -> None:
    run()

if __name__ == "__main__":
    main()