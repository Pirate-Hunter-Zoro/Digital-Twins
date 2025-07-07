# 🧠 How a Sentence Becomes a Vector in a Transformer Encoder

This guide walks through how Transformers convert a batch of sentences into dense vector representations (embeddings), step by step.

---

## Step 1: Tokenization

Each sentence is split into tokens and mapped to vocabulary indices.

**Example:**

```text

"I love ramen" → \["I", "love", "ramen"] → \[101, 1045, 2293, 7953, 102]

```

**Shape after tokenization:**

```text

(sequence\_length,)

```

---

## Step 2: Embedding Lookup

Each token ID is mapped to a vector using an embedding matrix.

**Embedding Matrix Shape:**

```text

(vocab\_size, hidden\_dim)

```

**Resulting Sentence Representation:**

```text

(sequence\_length, hidden\_dim)

```

Add batch dimension:

```text

(batch\_size, sequence\_length, hidden\_dim)

```

We also add **positional encodings** at this stage to preserve token order.

---

## Step 3: Transformer Layers (Self-Attention + Feedforward)

Each layer includes:

- **Multi-Head Self-Attention**:  
  Each token computes attention over all other tokens to contextualize itself.

- **Feedforward Network**:  
  Applies transformation to each token vector individually.

- **Layer Normalization & Residual Connections**

**Output Shape (unchanged):**

```text

(batch\_size, sequence\_length, hidden\_dim)

```

These layers are stacked `N` times (e.g. 12 layers in BERT-base).

---

## Step 4: Pooling (Sentence Embedding)

After all transformer layers, we reduce the sequence into a single vector per sentence.

**Pooling Methods:**

- **Mean Pooling**:  
  Average over all token embeddings.
  
- **CLS Token**:  
  Use the embedding of the first `[CLS]` token (common in BERT).

**Final Output:**

```text

(batch\_size, hidden\_dim)

```

---

## 📊 Shape Summary

| Stage                  | Shape                                      |
|------------------------|--------------------------------------------|
| Token IDs              | `(batch_size, sequence_length)`            |
| Embeddings             | `(batch_size, sequence_length, hidden_dim)`|
| After Self-Attention   | `(batch_size, sequence_length, hidden_dim)`|
| Sentence Embedding     | `(batch_size, hidden_dim)`                 |

---

## 💡 Notes

- The `sequence_length` is the number of tokens per sentence (after padding/truncation).
- `batch_size` is the number of sentences processed at once.
- Transformers are order-agnostic, so **positional encodings** are critical.
- Self-attention is what allows each token to contextualize itself with the rest of the sentence.

---

## 🧠 Token → Vector: The Embedding Matrix

### Step-by-step

1. **Each token is mapped to an integer ID**
   This is done via a tokenizer. For example:

   ```text
   "ramen" → token ID 7953
   ```

2. **There’s a learned matrix of shape:**

   ```text
   (vocab_size, hidden_dim)
   ```

   This is called the **embedding matrix**, often initialized randomly and then trained during pretraining (e.g., on next token prediction).

3. **To get the vector for a token**, you just index into this matrix:

   ```python
   embedding_vector = embedding_matrix[token_id]
   ```

   In PyTorch terms:

   ```python
   nn.Embedding(vocab_size, hidden_dim)
   ```

---

### 📦 What’s really happening

When you process a batch of token IDs like:

```text
[
  [101, 2293, 102],    # sentence 1
  [101, 3185, 102]     # sentence 2
]
```

You run them through the embedding layer and get:

```text
(batch_size, sequence_length, hidden_dim)
```

This entire operation is **just a lookup** — like a dictionary but optimized via matrix indexing.

---

### 🧪 Example

Let’s say:

- `vocab_size = 30,000`
- `hidden_dim = 768`

Then:

- Your embedding matrix is `30,000 × 768`
- Token ID 7953 gives you row 7953 in this matrix → a vector of shape `(768,)`

This is what gets passed to the transformer layers (after adding positional encodings).

---
