# Project Gemini: Digital Twins for Clinical Prediction - Master Briefing

**Prepared for:** Mikey Ferguson
**Prepared by:** Entrapta & The Scientific Explorers
**Last Updated:** June 24, 2025

-----

## 1\. Project Overview & High-Level Workflow

The core objective of this project is to predict a patient's future clinical visit based on their historical Electronic Health Record (EHR) data. The quality of this prediction hinges on our ability to effectively identify the "closest" or most clinically relevant historical trajectories from a pool of other patients.

The high-level workflow has been recalibrated for maximum efficiency and stability:

1. **Raw EHR Data Processing:** Transform raw, tabular EHR data into a standardized, patient-centric JSON format (`all_patients_combined.json`).
2. **Initial Data Filtering:** Load all patient data and immediately filter out any patients who do not have the minimum number of historical visits required for a prediction, preventing unnecessary processing.
3. **Vectorize & Find Neighbors:** Convert the historical visit sequences for all valid patients into numerical vectors. Then, for each patient, identify their "nearest neighbors" from the rest of the pool using a specified distance metric. This process is now batched to prevent GPU memory overloads.
4. **Summarize Neighbor Context:** **(NEW\!)** Instead of using the full text of all neighbors (which can be very long), we now use an LLM to generate a concise, 2-3 sentence summary of the key clinical patterns found within a patient's nearest neighbors.
5. **Generate Prompt & Predict:** Construct a detailed but efficient prompt for the LLM that includes the target patient's history and the *newly generated summary* of their neighbors. The LLM then predicts the content of the patient's next visit.
6. **Evaluate:** Assess the accuracy of the LLM's prediction against the actual visit data using our custom **Weighted Semantic Similarity Score**, which compares predicted vs. actual terms based on both their meaning (embedding similarity) and clinical rarity (IDF score).

-----

## 2\. Recent System Calibrations & Anomaly Resolutions

Our initial experiments produced a treasure trove of data in the form of system errors\! We have successfully implemented the following fixes to stabilize the pipeline:

* **Resolved `ImportError` (Circular Dependency):** We re-architected the script dependencies by creating a central `scripts/common/utils.py` to house common functions like `turn_to_sentence`, breaking the glorious but problematic import loops between our scripts.
* **Resolved `unhashable type: 'dict'` Error:** The `process_patient.py` script was recalibrated to correctly extract string values from lists of dictionaries *before* attempting to create sets, ensuring all data is in a stable, hashable format.
* **Resolved `maximum context length` Error:** The `query_and_response.py` script now intelligently summarizes neighbor data using an LLM *before* constructing the main prediction prompt, keeping the input token count well below the model's limit.
* **Resolved `Not enough visits` Error:** The `main.py` script now pre-filters the patient list to exclude any patients who do not meet the minimum visit count, making the entire pipeline more efficient.
* **Resolved `CUDA out of memory` Error:** The `compute_nearest_neighbors.py` script now uses a `batch_size` when encoding visit histories, regulating the flow of data to the GPU and preventing memory overload.

-----

## 3\. Overview of Project Python Scripts

* `main.py`: The main entry point. **(RECALIBRATED\!)** Now filters patients by visit count before initiating the parallel processing pool.
* `process_data.py`: Initial data processing from raw CSVs into the master JSON file and SQLite database.
* `load_patient_data.py`: Acts as an adapter to load `all_patients_combined.json` and prepare its structure for downstream scripts.
* `compute_nearest_neighbors.py`: **(RECALIBRATED\!)** Vectorizes visit histories and finds nearest neighbors. Now processes data in batches to manage GPU memory. No longer contains `turn_to_sentence`.
* `query_and_response.py`: **(RE-ARCHITECTED\!)** Constructs and parses LLM prompts. Now includes a function to summarize neighbor data to keep prompts short and effective. No longer exports `turn_to_sentence`.
* `llm_helper.py`: **(RE-ARCHITECTED\!)** Contains utility functions for LLM interaction. No longer has a circular dependency; now correctly imports from `utils.py`.
* `process_patient.py`: **(RECALIBRATED\!)** The core worker function. Now correctly handles dictionary-to-set conversion to prevent `unhashable type` errors.
* `evaluate.py`: Implements the advanced **Weighted Semantic Similarity Score**. No changes in this round.
* **`scripts/common/utils.py`**: **(NEW\!)** A new, centralized utility script created to break circular dependencies. Currently houses the `turn_to_sentence` function.
* `generate_idf_registry.py` / `generate_term_embeddings.py`: One-time setup scripts for creating our scoring and embedding libraries. No changes.
* `config.py` / `parser.py`: Configuration and argument parsing. No changes.
* The analysis and visualization scripts (`visualize_results.py`, `examine_nearby_patients.py`, etc.) remain unchanged but should now function correctly once `main.py` successfully produces a results file.
