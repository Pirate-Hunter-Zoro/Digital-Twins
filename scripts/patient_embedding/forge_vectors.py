"""Stage 2 entrypoint.
Loads Stage-1 narratives, extracts sections, embeds, and writes vectors.
CLI-only; delegates to stage2.runner.run()."""
from scripts.patient_embedding.stage2.runner import run  # type: ignore

def main() -> None:
    run()

if __name__ == "__main__":
    main()