# Digital Twins: Predictive Patient Modeling

## Project Phase: Experiment A – Representation Method Evaluation

---

### ✅ Completed Milestones

- **Integrated Full Evaluation Pipeline**  
  (`prepare_combined_embedding_terms.py`, `generate_combined_term_embeddings_mpnet.py`, and `run_debug_eval.py` now work end-to-end.)
- **Defaulted to Robust Embedder**  
  Due to incompatibilities with other models, `biobert-mnli-mednli` is now the default transformer.
- **Centralized Utilities and Config**  
  Pathing, config, and embedding logic now modularized for consistent experiment replication.

---

## 📍 Current Objectives: Experiment A

### 🎯 Immediate Goals

- [ ] **Monitor Slurm Jobs** (`run_visit_sentence.ssub`, `run_bag_of_codes.ssub`) and ensure they complete without failure.
- [ ] **Run `visualize_results.py`** to evaluate:
  - Visit-sentence-based representation
  - Bag-of-codes-based representation
- [ ] **Compare Score Distributions** to test Hypothesis H3.

---

## 🔭 Future Campaign: Embedding Evaluation, Metrics, Cleanup

### 🧪 Representation Variants

- [ ] **[R4] Temporal Embedding**: Add positional encodings to encode visit order.
- [ ] **[R5] Auto-summary Input**: Use LLM to summarize visit history pre-vectorization.

### 🚫 (Deprecate TF-IDF Option)
>
> All embeddings should use sentence transformer models going forward.

- [x] Remove TF-IDF logic from config and embedding flow.
- [x] Update validation to expect e.g. `'biobert-mnli-mednli'`, not `"tfidf"`.

---

## ✨ NEW: Digital Twin Ground Truth Refactors

- [ ] **Refactor Output Flow**  
  Ensure that:
  - `main.py` saves the *raw LLM predictions* only (no similarity scores).
  - Downstream evaluation (e.g., scoring, neighbor correlation) happens post-hoc via dedicated analysis scripts.
- [ ] **Decouple Evaluation**  
  Create explicit pipeline separation:
  
```text
Step 1: run main.py → produces raw predictions JSON
Step 2: run scoring scripts → attach scores + generate analysis
Step 3: run spearman analysis, semantic match audits, etc.
```

---

## 📎 Notes

- Config now stores `representation_method`, `vectorizer_method`, and `distance_metric` independently for full reproducibility.
- Embedded terms must be grouped by category (`diagnoses`, `medications`, `treatments`) before vectorization.
- Scoring scripts use cosine similarity + IDF greedy matching to assess LLM output vs ground truth.

---

Updated: June 27, 2025  
Maintainer: Mikey Ferguson 💡
