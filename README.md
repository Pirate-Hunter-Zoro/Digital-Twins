# SHAP Analysis for EHR Narrative Similarity

## 1. Project Objective

This project investigates the contribution of different EHR narrative sections to the overall semantic similarity of a patient's record. The goal is to quantify which sections—Summary, Medications, or Diagnostics—are most influential in determining the similarity between two complete patient narratives.

## 2. Data & Preprocessing

1.  **Input Data:** A collection of patient EHR narratives (as `.md` files).
2.  **Parsing:** Each narrative is parsed to extract text from three distinct sections:
    * `segment_narrative` (The "Patient Summary Narrative")
    * `segment_medications` (The "Medications" list)
    * `segment_diagnoses` (The "Diagnostics" list)
3.  **Components:** For each patient, four text components are stored: `full_text`, `segment_narrative`, `segment_medications`, and `segment_diagnoses`.

## 3. Methodology

Two parallel SHAP analyses are conducted. Both create a feature matrix `X` and a target vector `y` from all possible pairs of patients in the dataset.

* **Features (X):** A 3-dimensional vector representing the similarity of the parts:
    * `X_1`: Similarity of `segment_narrative`
    * `X_2`: Similarity of `segment_medications`
    * `X_3`: Similarity of `segment_diagnoses`
* **Target (y):** A scalar value representing the similarity of the whole:
    * `y`: Similarity of `full_text`

### Analysis A: Embedding-Based Similarity

This analysis uses vector embeddings (e.g., Qwen8b embedder) to calculate similarity.

* **X Features:**
    * `X_1 = cosine_similarity(embed(narrative_A), embed(narrative_B))`
    * `X_2 = cosine_similarity(embed(meds_A), embed(meds_B))`
    * `X_3 = cosine_similarity(embed(diags_A), embed(diags_B))`
* **y Target:**
    * `y = cosine_similarity(embed(full_text_A), embed(full_text_B))`

### Analysis B: LLM Judge-Based Similarity

This analysis uses an LLM (e.g., medgemma) to provide a direct semantic similarity score (0-1).

* **X Features:**
    * `X_1 = llm_judge(narrative_A, narrative_B)`
    * `X_2 = llm_judge(meds_A, meds_B)`
    * `X_3 = llm_judge(diags_A, diags_B)`
* **y Target:**
    * `y = llm_judge(full_text_A, full_text_B)`

## 4. Modeling and Explanation

For both Analysis A and B:

1.  **Model Training:** Two regression models (Linear Regression, Random Forest Regressor) are trained to predict `y` using the features `X`.
2.  **SHAP Explanation:** `shap.Explainer` (e.g., `LinearExplainer`, `TreeExplainer`) is used on the trained models and test data (`X_test`) to compute SHAP values.
3.  **Goal:** The primary output is the SHAP summary plot (bar chart), which shows the `mean(|SHAP value|)` for each feature. This value quantifies the average impact of each component's similarity on the model's prediction of the full text's similarity.