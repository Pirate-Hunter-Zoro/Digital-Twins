# June 26, 2025

- [x] **Integrated Full Evaluation Pipeline**: The three key scripts now work together reliably (`prepare_combined_embedding_terms.py`, `generate_combined_term_embeddings_mpnet.py`, and `run_debug_eval.py`).
- [x] **Defaulted to Working Embedder**: Due to multiple incompatibilities (HuggingFace SSL errors, `model_type` issues, numpy/tensor compatibility), we defaulted to `biobert-mnli-mednli` for now.
- [ ] **Flagged Future Cleanup**: Legacy or unused scripts will be identified and removed after confirming that all new components are stable and modularized.

## Project Gemini: TODO & Lab Notebook

**Last Updated:** June 24, 2025
**Current Phase:** Experimentation - Phase A (Representation Methods)

---

## I. Session Accomplishments: Pipeline Debugging and Stabilization

This session focused on diagnosing and resolving critical errors from the initial job run. The pipeline is now stable and has been upgraded to support modular experimentation.

- [x] **FIXED `unhashable type: 'dict'`:** Corrected data handling in `process_patient.py` to convert lists of dictionaries to lists of strings before creating sets.
- [x] **FIXED `ImportError` (Circular Dependency):** Re-architected script dependencies by creating a central `scripts/common/utils.py` for shared functions, breaking the import loops.
- [x] **FIXED `maximum context length exceeded`:** Implemented an LLM-based summarization step in `query_and_response.py` to condense neighbor data and reduce prompt token count.
- [x] **FIXED `CUDA out of memory`:** Integrated batch processing (`batch_size=32`) into the vector encoding step in `compute_nearest_neighbors.py` to manage GPU memory usage.
- [x] **FIXED `Not enough visits`:** Implemented a pre-filtering step in `main.py` to only process patients with a sufficient number of historical visits.
- [x] **FIXED `TypeError` in `main.py`:** Corrected the `setup_config` function call to properly pass the new `representation_method` argument.
- [x] **MODULARIZED for Experiment A:** Upgraded `parser.py`, `config.py`, and `compute_nearest_neighbors.py` to support selectable patient history representation methods.
- [x] **CREATED LAUNCH SCRIPTS:** Developed robust `.ssub` scripts for submitting experiments to the Slurm cluster, including vLLM server initialization and cleanup.

---

## II. Next Steps: Experimental Execution

The system is stable. The experiments are configured and running. The next phase focuses on data collection and analysis.

### Immediate Objective: Complete Experiment A

- [ ] **Monitor Slurm Jobs:** Observe the two running jobs (`run_visit_sentence.ssub` and `run_bag_of_codes.ssub`) to ensure they complete successfully.
- [ ] **Analyze Results:** Once jobs are complete, use `visualize_results.py` on both output files (`patient_results_*_visit_sentence_*.json` and `patient_results_*_bag_of_codes_*.json`).
- [ ] **Compare and Conclude (H3):** Compare the score distributions from both runs to determine which representation method yields more accurate predictions, addressing Hypothesis H3 from the project plan.

### Future Experiments (The Grand Campaign)

- [ ] **Expand Experiment A (Representations):**
  - [ ] Implement **R1: Baseline TF-IDF**. This requires adding a 'tfidf' option to the vectorizer and updating the logic to use `TfidfVectorizer` instead of a sentence transformer.
  - [ ] Implement **R4: Temporal Embedding**. This will involve modifying the vectorization process to include positional encodings.
  - [ ] Implement **R5: Auto-summary**. This involves using an LLM to summarize visit histories *before* vectorization.
- [ ] **Initiate Experiment B (Embeddings):** Once the best representation method is identified, begin testing it with the different embedding models outlined in the project plan (E1-E6).
- [ ] **Initiate Experiment C (Metrics):** After finding the best representation/embedding combination, test it with different distance metrics, including the planned "Learned Metric" using a Siamese network.
