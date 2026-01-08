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

### 2. Stage 1: Narrative Generation (`scripts/digital_twins/narratives`)
Transforms the structured JSONs into textual narratives.
* **`generator.py`**: Iterates through the cohort, applies the `deterministic_narrative` logic, and saves `.md` files to the `DETERMINISTIC_NARRATIVES_DIR`.
* **`runner.py`**: Orchestrates the generation job via Slurm.

### 3. Stage 2: Vector Embedding (`scripts/digital_twins/embeddings`)
**The Forge.** Converts text narratives into high-dimensional vectors using the `StringEmbedder`.
* **`forge_vectors.py`**: The main driver.
    1.  Reads `.md` files from Stage 1.
    2.  Batches them (CPU-bound or GPU-bound).
    3.  Feeds them to the `StringEmbedder`.
* **Artifacts**: This stage populates the SQLite database `vectors.db`.

### 4. Stage 3: Retrieval & Inference (`scripts/digital_twins/retrieval`)
**The Judge.** Finds and scores patient similarity.
* **`retriever.py`**:
    * Loads the entire `vectors.db` into memory as a normalized matrix.
    * Performs fast cosine similarity search (Pre-filter) to find top-K candidates.
* **`scorer.py`**:
    * The LLM Judge. Takes candidate pairs and evaluates clinical similarity using a rigid JSON schema.
    * **Caching**: Stores expensive LLM outputs in `judgements.db` (Table: `llm_judgements`) to prevent redundant inference.
    * **Logic**: Checks cache -> Formats Prompt -> Calls vLLM -> Parses JSON -> Saves Result.
    * **Critical**: This script enforces strict validation. If the LLM returns malformed JSON, the process raises an exception and halts to prevent data corruption.
* **`pipeline_runner.py`**: (Coming Soon) Orchestrates the end-to-end flow: `Index Patient -> Vector Search -> Top Candidates -> LLM Scoring -> Final Report`.

### 5. Models (`scripts/models`)
Interfaces for the neural networks.
* **`patient_embedder.py`**:
    * Wraps `SentenceTransformer` (e.g., Qwen).
    * **Storage**: Manages a SQLite connection to `vectors.db`.
    * **Logic**: Checks the DB for existing IDs (MD5 hash of text). If missing, computes the embedding and inserts it as a binary BLOB.
    * **Scrubbing**: Respects `SCRUB_VECTORS` env var to force re-computation.
* **`vllm_client.py`**: Client for interacting with the vLLM inference server (for LLM-based narrative generation).

### 6. Shared Utilities (`scripts/shared`)
* **`similarity.py`**: **The Search Engine.**
    * Computes Cosine Similarity between patient IDs.
    * **Caching**: Uses the `similarities` table in `vectors.db` to store `(id_a, id_b, score)`.
    * **Efficiency**: Checks the DB cache first. If a miss, loads vector BLOBs, computes dot product, and saves the result.
* **`utils.py`**: Core helpers (hashing logic `generate_string_id`, etc.).
* **`io.py`**: Standardized file handling.

---

## The Vault: Databases

### Vector Storage (`vectors.db`)
Located at `ARTIFACTS_DIR/vectors.db` (specific path depends on embedding model). Handles scale without file system overhead.

**Table: `vectors`**
Stores the raw embeddings.
| Column | Type | Description |
| :--- | :--- | :--- |
| `id` | `TEXT (PK)` | MD5 Hash of the narrative text. |
| `patient_id` | `TEXT` | Patient ID of the corresponding narrative. |
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

### Judgement Storage (`judgements.db`)
Located at `JUDGEMENTS_DIR`.

**Table: `llm_judgements`**
Caches the expensive qualitative evaluations from the LLM.
| Column | Type | Description |
| :--- | :--- | :--- |
| `id_a` | `TEXT (PK)` | First Patient ID. |
| `id_b` | `TEXT (PK)` | Second Patient ID. |
| `overall_score` | `INTEGER` | Numeric Similarity Score |
| `full_response` | `TEXT`| Full Response from LLM Including Justification |

---

## Configuration

The pipeline requires a `.env` file. Below are the standard configurations:

### General & Reproducibility
* `SEED`: 42 (Ensures deterministic behavior).

### Data Paths (Input)
* `PREP_DATA_DIR`: `/media/studies/ehr_study/data-EHR-prepped/DV250901v1-PV251208v1/PrepData/`
* `PROCEDURE_CSV_PATH`: Path to `Procedure_Table.csv`
* `MEDICATION_CSV_PATH`: Path to `Medication_Table.csv`
* `DIAGNOSIS_CSV_PATH`: Path to `Diagnosis_Table.csv`
* `ENCOUNTER_CSV_PATH`: Path to `Encounter_Table.csv`
* `PERSON_CSV_PATH`: Path to `Person_Table.csv`

### Analysis & Cohorts
* `ANALYSIS_DIR`: `/media/studies/ehr_study/analysis/mferguson/`
* `MDD_MED_DATE_CSV_PATH`: `${ANALYSIS_DIR}/post_mdd_ad_index.csv`
* `COHORT_PATH`: `${ANALYSIS_DIR}/mdd_only_patients.csv`
* `PATIENT_JSON_DIR`: `${ANALYSIS_DIR}/patient_json`
* `SLICED_PATIENT_JSON_DIR`: `${ANALYSIS_DIR}/sliced_patient_json`

### Models

**Embedding Model**
* `EMBEDDER_MODEL_NAME`: `Qwen-Qwen3-Embedding-8B`
* `EMBEDDER_MODEL_PATH`: `${ANALYSIS_DIR}/models/${EMBEDDER_MODEL_NAME}`
* `EMBEDDER_DEVICE`: `cpu` (Avoid `cuda` for large strings to prevent OOM).
* `EMBEDDER_BATCH_SIZE`: 32

**Generative Model (vLLM)**
* `VLLM_MODEL_NAME`: `google_medgemma-27b-text-it`
* `VLLM_URL`: `http://compute306:8000`
* `MAX_TOKENS`: 8192

### Artifacts & Output
* `ARTIFACTS_DIR`: `${ANALYSIS_DIR}/artifacts/`
* `VECTORS_DIR`: `${ARTIFACTS_DIR}/${EMBEDDER_MODEL_NAME}/`
* `JUDGEMENTS_DIR`: `${ARTIFACTS_DIR}/${VLLM_MODEL_NAME}/`
* `SCRUB_VECTORS`: 0 (Set to 1 to force re-embedding).

### Concurrency
* `NUM_WORKERS_NON_LLM_TASK`: 16
* `NUM_WORKERS_LLM_TASK`: 2 (Keep low to avoid overwhelming vLLM).

## Usage

**To Forge Vectors (Stage 2):**
```bash
python -m scripts.digital_twins.embeddings.forge_vectors

```

**To Calculate Similarity:**

```python
from scripts.shared.similarity import cosine
score = cosine("patient_id_A", "patient_id_B")

```
