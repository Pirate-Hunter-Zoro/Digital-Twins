import argparse
from sentence_transformers import SentenceTransformer
import os

def read_model_ids(file_path):
    with open(file_path, "r") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

def test_model_compatibility(model_id):
    try:
        print(f"🔍 Testing model: {model_id}")
        model = SentenceTransformer(model_id)
        print(f"✅ Success: {model_id}")
        return True
    except Exception as e:
        print(f"❌ Failed: {model_id}")
        print(f"   ↳ Error: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Test embedding model compatibility with sentence-transformers.")
    parser.add_argument('--file', type=str, required=True, help="Path to file containing model IDs.")
    args = parser.parse_args()

    model_file = args.file
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"🚫 File not found: {model_file}")

    model_ids = read_model_ids(model_file)
    print(f"🧪 Testing {len(model_ids)} models from {model_file}")
    for model_id in model_ids:
        test_model_compatibility(model_id)

if __name__ == "__main__":
    main()
