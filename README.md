# Digital Twins: Patient Representation & Embedding Pipeline

This repository contains the pipeline for converting Electronic Health Record (EHR) data into "Digital Twins"—vectorized representations of patient narratives capable of semantic search and cohort analysis. 

The pipeline is divided into **Data Loading**, **Stage 1 (Narrative Generation)**, and **Stage 2 (Vector Embedding)**.

## Project Structure

### 1. Data Loading (`scripts/data_loading`)
The foundation. These scripts ingest raw EHR exports and structure them into usable patient objects.
* **`build_jsons.py`**: The initial ETL step. Converts raw CSVs/SQL dumps into per-patient JSON files.
* **`create_cohort.py`**: Filters the total population down to the study cohort (e.g., MDD patients).
* **`deterministic_narrative.py`**: The logic that deterministically translates structured JSON features (labs, meds, diagnoses) into a human-readable Markdown narrative.
* **`features.py`**: Extractors for specific clinical features.
* **Definitions**: `diagnoses_definitions.py`, `med_definitions.py`, etc., map codes to clinical text.

### 2. Stage 1: Narrative Generation (`scripts/digital_twins/stage1`)
Transforms the structured JSONs into textual narratives.
* **`generator.py`**: Iterates through the cohort, applies the `deterministic_narrative` logic, and saves `.md` files to the `DETERMINISTIC_NARRATIVES_DIR`.
* **`runner.py`**: Orchestrates the generation job via Slurm.

### 3. Stage 2: Vector Embedding (`scripts/digital_twins/stage2`)
**The Forge.** Converts text narratives into high-dimensional vectors using the `StringEmbedder`.
* **`forge_vectors.py`**: The main driver. 
    1.  Reads `.md` files from Stage 1.
    2.  Batches them (CPU-bound or GPU-bound).
    3.  Feeds them to the `StringEmbedder`.
* **Artifacts**: This stage populates the SQLite database `vectors.db`.

### 4. Models (`scripts/models`)
Interfaces for the neural networks.
* **`string_embedder.py`**: 
    * Wraps `SentenceTransformer` (e.g., Qwen).
    * **Storage**: Manages a SQLite connection to `vectors.db`.
    * **Logic**: Checks the DB for existing IDs (MD5 hash of text). If missing, computes the embedding and inserts it as a binary BLOB.
    * **Scrubbing**: Respects `SCRUB_VECTORS` env var to force re-computation.
* **`vllm_client.py`**: Client for interacting with the vLLM inference server (for LLM-based narrative generation).

### 5. Shared Utilities (`scripts/shared`)
* **`similarity.py`**: **The Search Engine.**
    * Computes Cosine Similarity between patient IDs.
    * **Caching**: Uses the `similarities` table in `vectors.db` to store `(id_a, id_b, score)`.
    * **Efficiency**: Checks the DB cache first. If a miss, loads vector BLOBs, computes dot product, and saves the result.
* **`utils.py`**: Core helpers (hashing logic `generate_string_id`, etc.).
* **`io.py`**: Standardized file handling.

---

## The Vault: `vectors.db`

The embedding system relies on a single SQLite database located at `VECTORS_DIR/vectors.db` to handle the scale of 50,000+ patients without file system overhead.

### Schema

**Table: `vectors`**
Stores the raw embeddings.
| Column | Type | Description |
| :--- | :--- | :--- |
| `id` | `TEXT (PK)` | MD5 Hash of the narrative text. |
| `vector` | `BLOB` | The numpy array (`float32`) serialized to bytes. |
| `text` | `TEXT` | The raw narrative text (for audit/retrieval). |
| `length` | `INTEGER` | Character count of the text. |

**Table: `similarities`**
Caches comparison scores to avoid re-computing dot products (billions of potential pairs).
| Column | Type | Description |
| :--- | :--- | :--- |
| `id_a` | `TEXT (PK)` | First Patient ID (alphabetically sorted). |
| `id_b` | `TEXT (PK)` | Second Patient ID. |
| `score` | `REAL` | Cosine similarity (0.0 - 1.0). |

*Note: The `similarities` table includes an index on `score` for rapid outlier analysis.*

---

## Configuration (.env)

The pipeline requires the following environment variables:

```bash
# Directories
VECTORS_DIR="/path/to/storage/vectors"
SLICED_PATIENT_JSON_DIR="/path/to/data/sliced_jsons"
UNSLICED_PATIENT_JSON_DIR="/path/to/data/unsliced_jsons"
DETERMINISTIC_NARRATIVES_DIR="/path/to/data/narratives_md"

# Model Configuration
EMBEDDER_MODEL_PATH="/path/to/local/model/weights"
EMBEDDER_MODEL_NAME="Qwen-Embedding-Checkpoint"
EMBEDDER_DEVICE="cpu" # or "cuda"
EMBEDDER_BATCH_SIZE="32"

# Operations
SCRUB_VECTORS="0"      # Set to "1" to force re-embedding
SCRUB_SIMILARITY="0"   # Set to "1" to force re-calculation of scores

```

## Usage

**To Forge Vectors (Stage 2):**

```bash
python -m scripts.digital_twins.stage2.forge_vectors

```

**To Calculate Similarity:**

```python
from scripts.shared.similarity import cosine
score = cosine("patient_id_A", "patient_id_B")

```
