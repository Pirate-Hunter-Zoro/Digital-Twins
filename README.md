# Digital Twins: Patient Representation & Embedding Pipeline

This repository contains the pipeline for converting Electronic Health Record (EHR) data into "Digital Twins"—vectorized representations of patient narratives capable of semantic search, cohort analysis, and clinical outcome prediction.

The pipeline is divided into **Data Loading**, **Stage 1 (Narrative Generation)**, **Stage 2 (Vector Embedding)**, **Stage 3 (Neighbor Retrieval)**, and **Stage 4 (Prediction)**.

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
**The Forge.** Converts text narratives into high-dimensional vectors using the `PatientEmbedder`.
* **`forge_vectors.py`**: The main driver.
    1.  Reads `.md` files from Stage 1.
    2.  Batches them.
    3.  Feeds them to the `PatientEmbedder` to generate embeddings.
* **Artifacts**: This stage populates the SQLite database `vectors.db`.

### 4. Stage 3: Retrieval & Scoring (`scripts/digital_twins/neighbors`)
**The Judge.** Finds and scores patient similarity.
* **`retriever.py`**:
    * Loads the entire `vectors.db` into memory as a normalized matrix.
    * Performs fast cosine similarity search (Pre-filter) to find top-K candidates.
    * **Self-Exclusion**: Implements logic to exclude specific IDs from search results (essential for backtesting).
    * **Note**: Handles retrieval of raw narratives and mapping of hashed IDs back to Patient IDs.
* **`scorer.py`**:
    * The LLM Judge. Takes candidate pairs and evaluates clinical similarity using a rigid JSON schema.
    * **Caching**: Stores expensive LLM outputs in `judgements.db` (Table: `llm_judgements`) to prevent redundant inference.
    * **Logic**: Checks cache -> Formats Prompt -> Calls vLLM -> Parses JSON -> Saves Result.

### 5. Stage 4: Prediction & Evaluation (`scripts/digital_twins/predictions`)
**The Oracle.** Uses the retrieved neighbors to predict clinical outcomes.
* **`trd_predictor.py`**:
    * Implements the **Digital Twin Matcher** logic for Treatment-Resistant Depression (TRD).
    * **Workflow**:
        1.  Retrieves top-K neighbors via `retriever.py` (excluding the query patient).
        2.  Scores neighbors via `scorer.py`.
        3.  Applies exponential weighting ($w = score^\alpha$) to emphasize strong matches.
        4.  Computes weighted probability of TRD risk ($P(TRD)$).
    * **Safety**: Calculates Effective Sample Size (ESS) to flag low-confidence predictions.
* **`evaluate_trd.py`**:
    * Backtesting script.
    * **Sampling**: Selects a balanced random sample of TRD-positive and TRD-negative patients from the vector database.
    * **Metrics**: Computes **ROC AUC** (Discrimination), **Brier Score** (Calibration), and **Mean ESS** (Confidence).
    * **Output**: Saves a summary text file and a full CSV log of every prediction (`trd_evaluation_results.csv`).

### 6. Models (`scripts/models`)
Interfaces for the neural networks.
* **`patient_embedder.py`**:
    * Wraps `SentenceTransformer` (e.g., Qwen).
    * **Storage**: Manages a SQLite connection to `vectors.db`.
    * **Logic**: Checks the DB for existing IDs (MD5 hash of text). If missing, computes the embedding and inserts it as a binary BLOB.
    * **Scrubbing**: Respects `SCRUB_VECTORS` env var to force re-computation.
* **`vllm_client.py`**: Client for interacting with the vLLM inference server (for LLM-based narrative generation or scoring).

### 7. Shared Utilities (`scripts/shared`)
* **`similarity.py`**: **The Search Engine.**
    * Computes Cosine Similarity between patient IDs.
    * **Caching**: Uses the `similarities` table in `vectors.db`.
* **`utils.py`**: Core helpers (hashing logic `generate_string_id`, etc.).
* **`io.py`**: Standardized file handling.

---

## The Vault: Databases

### Vector Storage (`vectors.db`)
Located at `ARTIFACTS_DIR/vectors.db`.

**Table: `vectors`**
Stores the raw embeddings.
| Column | Type | Description |
| :--- | :--- | :--- |
| `id` | `TEXT (PK)` | MD5 Hash of the narrative text. |
| `patient_id` | `TEXT` | Patient ID of the corresponding narrative. |
| `vector` | `BLOB` | The numpy array (`float32`) serialized to bytes. |
| `text` | `TEXT` | The raw narrative text (for audit/retrieval). |
| `length` | `INTEGER` | Character count of the text. |

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
* `WEIGHTING_EXPONENT`: 4.0 (Alpha value for weighting similarity scores in TRD prediction).
* `TRD_TEST_COUNT`: 50 (Number of patients to sample for evaluation).

### Data Paths (Input)
* `TRD_LIST_PATH`: Path to the text file containing IDs of TRD-positive patients.
* `PREP_DATA_DIR`: Raw data directory.
* `COHORT_PATH`: `${ANALYSIS_DIR}/mdd_only_patients.csv`

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
* `RESULTS_DIR`: `${ARTIFACTS_DIR}/${EMBEDDER_MODEL_NAME}/${VLLM_MODEL_NAME}/`

### Retrieval & Search
* `NUM_NEIGHBOR_PATIENTS`: 200 (Number of candidates to retrieve before LLM scoring).

## Usage

**To Forge Vectors (Stage 2):**
```bash
python -m scripts.digital_twins.embeddings.forge_vectors

```

**To Evaluate TRD Prediction (Stage 4):**

```bash
python -m scripts.digital_twins.predictions.evaluate_trd

```
