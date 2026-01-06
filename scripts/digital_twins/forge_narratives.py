"""Stage 1 entrypoint.
Delegates to narratives.runner.run() without changing behavior."""
from scripts.digital_twins.narratives.runner import run  # type: ignore

def main() -> None:
    run()

if __name__ == "__main__":
    main()