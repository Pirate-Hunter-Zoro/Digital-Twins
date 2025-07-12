# Running vLLM with Qwen3-30B on LIBR Cluster

This guide walks you through launching `vllm` on a compute node, using a local Conda environment, and testing the model with the OpenAI-compatible API.

---

## 1. Set up your Conda environment

Use Python 3.10–3.12. Run the following:

```bash
conda create -n venv python=3.12 -y
conda activate dt_env
pip install -r requirements.txt
```

Your `requirements.txt` should include at minimum:

```
vllm
torch
transformers
openai
```

---

## 2. Launch the vLLM server on a compute node

From `submit0`, request a GPU compute node:

```bash
srun --partition=c3_short --gres=gpu --pty bash
```

Once you're in the compute node (e.g. `compute305`):
```bash
conda activate dt_env
```

Obtain the snapshot of the model you have downloaded from running ```download_model.py```
```bash
vllm serve unsloth/medgemma-27b-text-it-bnb-4bit \
  --dtype float16 \
  --served-model-name medgemma \
  --gpu-memory-utilization 0.5 \
  --host 0.0.0.0 \
  --max-model-len 5000
```

Wait for the server to fully load. You should see a message like:

```
INFO:     Started server process [xxxxx]
INFO:     Application startup complete.
```

---

## 3. From a second terminal, run your query script

In a second terminal, SSH to `submit0` and then SSH to the same compute node (do not request a GPU again).

```bash
ssh compute305
```

Then:

```bash
conda activate dt_env
python query_llm.py
```

---

## `query_llm.py` (example)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",  # or replace with internal IP if remote access
    api_key="not-needed-for-localhost",   # any non-empty string to avoid Bearer header errors
)

response = client.chat.completions.create(
    model="cognitivecomputations/Qwen3-30B-A3B-AWQ",  # Must match name in `vllm serve`
    messages=[{"role": "user", "content": "What is the capital of France?"}],
    temperature=0.7,
)

print(response.choices[0].message.content)
```

---

## Summary

- Launch `vllm serve` on a compute node.
- SSH into that exact node from another terminal to make API calls.
- Make sure both terminals use the same Conda environment.

Happy inferencing!