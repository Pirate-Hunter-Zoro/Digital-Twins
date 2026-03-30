from scripts.digital_twins.vectors.generator import generate_deterministic_vectors

def run() -> None:
    print("Running deterministic vector generation...", flush=True)
    generate_deterministic_vectors()
    print("[Vectors] complete.", flush=True)