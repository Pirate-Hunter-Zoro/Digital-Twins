# Digital Twins LLM Prediction Evaluation Pipeline

This README outlines the pipeline, structure, and next steps for evaluating patient visit prediction quality using sentence transformers and LLM-generated predictions.

---

## ✅ Current Functionality

1. **Config Setup**

   - Dynamic configuration handled via `config.py` using `setup_config()` and `get_global_config()`.
   - All config references now expect vectorizer models like `biobert-mnli-mednli` (no more TF-IDF logic).

2. **Term Cleaning & Embedding**

   - `prepare_categorized_embedding_terms.py`: parses prediction JSONs and saves a categorized term list.
   - `generate_term_embeddings.py`: loads the categorized term list and uses sentence transformers to generate category-wise normalized embeddings.

3. **LLM Prediction**

   - `main.py`: Generates LLM predictions for patient visit sequences.
   - Configurable via `ProjectConfig`, soon to be CLI-compatible for SLURM batch job variation.

4. **Evaluation**

   - `evaluate.py`: Compares predicted vs actual terms using greedy cosine similarity + geometric mean of IDF weights.
   - `calculate_spearmans_rho.py`: Correlates LLM-judged neighbor relevance with Mahalanobis distance.

5. **Analysis & Diagnostics**

   - `analyze_embedding_cosine_stats.py`: Visualizes cosine similarity distributions.
   - `check_semantic_similarity.py`: Verifies semantically similar but lexically different terms.
   - `check_missing_embedding_terms.py`: Checks for embedding coverage issues.
   - `visualize_results.py`: Generates visualizations for evaluation metrics.

---

### 🧠 Next Steps

#### 🔧 Core Improvements

- Refactor all config references to expect sentence transformer vectorizers (e.g. `biobert-mnli-mednli`)
- Ensure `main.py` saves predictions *before* scores are calculated
- Defer evaluation scoring to post-processing stage

#### 🧪 Post-Prediction Evaluation Pipeline

- Run `evaluate.py` on `patient_results_*.json` to assign similarity scores
- Use `calculate_spearmans_rho.py` to analyze neighbor relevance vs mahalanobis distances
- Confirm alignment of pipeline with mentor feedback goals

#### 📊 Embedding Quality Exploration

- Identify pairs of terms with low, medium, and high cosine similarity
- Investigate cosine similarity distribution across term types
- Expand `check_semantic_similarity.py` for more pair validation examples

#### 🧹 Code Quality & Structure

- Refactor Slurm job compatibility for configurable parameter sets
- Modularize `main.py` and `evaluate.py` for easier testing
- Add logging where currently `print()` is used

---

### 🌀 System Overview Diagram (Text Mode)

```text

\[Patient JSON] --> \[main.py] --> patient\_results\_\*.json
|
v
\[prepare\_categorized\_embedding\_terms.py]
|
v
\[generate\_term\_embeddings.py] --> term\_embedding\_library\_by\_category.pkl
|
v
\[evaluate.py] --> Adds scores to prediction JSON
|
v
\[calculate\_spearmans\_rho.py] --> Spearman correlation CSV

```

---

🧪 Built with sentence-transformers. Fueled by science. Debugged with chaos. Powered by... coffee?

**Let's keep pushing science forward!** 💜

---

### 🧠 Updated Workflow Additions

#### 🔍 Nearest Neighbor Analysis (Pre-Evaluation)

We've introduced a dedicated SLURM job for computing nearest neighbors based on patient history vectors, allowing us to:

- Interpret digital twin prediction context
- Analyze local patient distributions (Mahalanobis)
- Visualize patterns and outliers

This step is **independent** and can be run **before or after** LLM predictions, depending on investigative goals.

Scripts:

- `compute_nearest_neighbors.py`
- SLURM templates: `submit_neighbors_bag_of_codes.ssub`, `submit_neighbors_visit_sentence.ssub`

Output:

- `data/neighbors_<config>.pkl`
