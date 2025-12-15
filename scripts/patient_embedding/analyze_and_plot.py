"""Stage 3 entrypoint.
Builds pairs, judges once, computes cosines, plots, and emits discordant bundles.
CLI-only; delegates to stage3.runner.run()."""
from __future__ import annotations

from scripts.patient_embedding.stage3.runner import run  # type: ignore

def main() -> None:
    run()

if __name__ == "__main__":
    main()