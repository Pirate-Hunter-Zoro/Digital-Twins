from scripts.digital_twins.vectors.generator import generate_feature_vectors

def run() -> None:
    print("Running feature vector generation...", flush=True)
    generate_feature_vectors()
    print("[Vectors] complete.", flush=True)