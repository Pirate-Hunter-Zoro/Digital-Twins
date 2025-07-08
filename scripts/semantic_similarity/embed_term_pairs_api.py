import argparse
import json
import os
from pathlib import Path
from dotenv import load_dotenv
import requests
from tqdm import tqdm

# 🔄 Load .env from root project directory
project_root = Path(__file__).resolve().parents[2]
load_dotenv(dotenv_path=project_root / ".env")

HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
if not HUGGINGFACE_TOKEN:
    raise EnvironmentError("HUGGINGFACE_TOKEN is not set in environment variables.")

HF_API_URL = "https://api-inference.huggingface.co/embeddings"

def get_embeddings(model_name, texts):
    headers = {
        "Authorization": f"Bearer {HUGGINGFACE_TOKEN}",
        "Content-Type": "application/json"
    }

    body = {"inputs": texts, "model": model_name}
    response = requests.post(HF_API_URL, headers=headers, json=body)

    if response.status_code != 200:
        raise RuntimeError(f"Request failed ({response.status_code}): {response.text}")

    return response.json()["embeddings"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Hugging Face model ID to use for embedding")
    args = parser.parse_args()

    model_name = args.model
    print(f"🔍 Using API-based model: {model_name}")

    term_pairs_path = project_root / "data" / "term_pairs.json"
    output_path = project_root / "data" / "embeddings" / f"{model_name.replace('/', '-')}.json"

    with open(term_pairs_path) as f:
        term_pairs = json.load(f)

    all_terms = sorted({term for pair in term_pairs for term in pair})
    print(f"🧠 Embedding {len(all_terms)} unique terms")

    embeddings = {}
    batch_size = 10

    for i in tqdm(range(0, len(all_terms), batch_size)):
        batch = all_terms[i:i + batch_size]
        embs = get_embeddings(model_name, batch)
        for term, emb in zip(batch, embs):
            embeddings[term] = emb

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(embeddings, f)

    print(f"✅ Done. Saved to {output_path}")

if __name__ == "__main__":
    main()
