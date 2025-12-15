"""Stage 1 entrypoint.
Delegates to stage1.runner.run() without changing behavior."""
from scripts.patient_embedding.stage1.runner import run  # type: ignore

def main() -> None:
    run()

if __name__ == "__main__":
    main()