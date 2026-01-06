"""Stage 2 entrypoint.
Loads Stage-1 narratives, extracts sections, embeds, and writes vectors.
CLI-only; delegates to embeddings.runner.run()."""
from scripts.digital_twins.embeddings.runner import run  # type: ignore

def main() -> None:
    run()

if __name__ == "__main__":
    main()