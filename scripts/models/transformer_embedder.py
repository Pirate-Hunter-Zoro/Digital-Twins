from transformers import AutoModel, AutoTokenizer
import torch
import numpy as np

class TransformerEmbedder:
    def __init__(self, model_path):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
            self.model = AutoModel.from_pretrained(model_path, local_files_only=True)
        except Exception as e:
            raise RuntimeError(f"❌ Failed to load model from {model_path}: {e}")

    def __call__(self, text: str):
        return self.encode(text)

    def encode(self, sentence: str, convert_to_numpy=True, normalize_embeddings=True):
        inputs = self.tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            output = self.model(**inputs)

        if hasattr(output, "last_hidden_state"):
            embeddings = output.last_hidden_state.mean(dim=1)
        elif hasattr(output, "pooler_output"):
            embeddings = output.pooler_output
        else:
            raise ValueError("Unsupported model output format")

        if normalize_embeddings:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        if convert_to_numpy:
            return embeddings.cpu().numpy()[0]
        return embeddings

    @property
    def embed_type(self):
        return "transformers-fallback"