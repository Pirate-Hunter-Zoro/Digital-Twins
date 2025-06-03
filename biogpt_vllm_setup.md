
# BioGPT Local Model Setup with vLLM (Manual Snapshot Style)

This guide documents the steps taken to download, transfer, and serve the `microsoft/BioGPT` model on the LIBR `submit0` cluster using vLLM.

---

## 📥 1. Download the Model on Local Machine

Visit: [https://huggingface.co/microsoft/biogpt/tree/main](https://huggingface.co/microsoft/biogpt/tree/main)

Download the following files:
- `pytorch_model.bin`
- `config.json`
- `vocab.json`
- `merges.txt`

---

## 🚀 2. Transfer Files to Submit0 Server

From your local machine:

```bash
scp config.json merges.txt vocab.json pytorch_model.bin \
mferguson@submit0:/home/librad.laureateinstitute.org/mferguson/models/Microsoft-BioGPT
```

---

## 🧱 3. Mimic Hugging Face Snapshot Directory

```bash
mkdir -p ~/.cache/huggingface/hub/models--microsoft--biogpt/snapshots
cp -r /home/librad.laureateinstitute.org/mferguson/models/Microsoft-BioGPT \
    ~/.cache/huggingface/hub/models--microsoft--biogpt/snapshots/manual
```

Create a symlink for cleaner access:

```bash
mkdir -p ~/models
ln -s ~/.cache/huggingface/hub/models--microsoft--biogpt/snapshots/manual ~/models/biogpt
```

---

## 🌐 4. Serve BioGPT Using vLLM

Activate your environment and launch the server:

```bash
conda activate vllm_env

vllm serve ~/models/biogpt \
  --dtype float16 \
  --served-model-name biogpt
```

---

## ✅ Test the Endpoint (Optional)

```bash
curl http://localhost:8000/v1/models
```

You should see `biogpt` listed in the output.

---

You're now ready to use `BioGPT` in your OpenAI-compatible apps!
