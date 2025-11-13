# The Digital Twin Stack: A Causal Implementation

## 1. The Core Philosophy: From Statistical to Causal Twins

This project moves beyond finding mere "statistical twins" based on superficial similarity. The fundamental goal is to construct a "causal twin" by using a rigorous statistical backbone.  We are not just asking "who looks like this patient?" but rather, "what would happen to this patient under a different course of action, accounting for all confounding factors?"  This is achieved by integrating a propensity-weighted causal layer at every step, transforming simple retrieval into a robust "what-if" engine. 

---

## 2. The Foundation: A Disciplined and Reproducible Workflow

Before any analysis, a rigid structure is established to ensure perfect reproducibility and prevent "data creep." 

1.  **Directory Structure:** A predictable folder layout is created to house all artifacts, from raw data to final models, ensuring any team member can navigate the project. 
2.  **Cohort Freeze:** The patient population (72,168 individuals) is defined and locked on day one.  This guarantees that all subsequent model improvements are due to the methods, not a shifting dataset. 
3.  **Deterministic Serializer:** Patient histories are converted into a standardized, human-readable Markdown format.  Each visit becomes a consistent line of text, ensuring that the same patient record always produces the exact same input for the models. 

---

## 3. The Causal Engine: Building the Backbone

This is the critical layer that separates this plan from simple similarity searching.

1.  **Propensity Scores:** An XGBoost model is trained to calculate the probability that a patient would receive a certain treatment (e.g., a medication switch) based on their history and characteristics.  This creates inverse-propensity weights (IPW) that can be used to balance the treatment and control groups, mimicking a randomized trial. 
2.  **Causal Forest:** A causal forest model is then trained, using the patient embeddings as input.  This model estimates the Individualized Treatment Effect (ITE) by predicting outcomes under both possible actions (e.g., "switch" vs. "stay"). 

---

## 4. The Two Paths to Prediction

With the causal foundation in place, two retrieval methods are employed to find twins and predict outcomes.

### Method 1: The Encoder Path (Causally-Adjusted k-Nearest Neighbors)

1.  **Embedding:** A BEHRT model creates a mathematical signature for each patient. 
2.  **Retrieval:** FAISS is used to find a set of nearest neighbors (e.g., k=50) for a target patient. 
3.  **The Verdict:** The prediction is **not** a simple average. An Augmented Inverse Propensity Weighting (AIPW) estimator is applied to this group of 50 neighbors, using the pre-computed IPW weights.  This provides a causally-adjusted risk estimate with confidence intervals, such as "Switch → TRD 18% (±4%)". 

### Method 2: The LLM Path (Clinician-Expert with Guardrails)

1.  **Pre-Filtering:** The encoder first identifies the 1,000 closest neighbors to narrow the search space. 
2.  **The Model's Judgment:** A fine-tuned Mistral-7B model, acting as a clinical expert, reviews the target patient and the 1,000 candidates to select the single best match. 
3.  **The Verdict:** The LLM generates a structured JSON output, including its chosen twin, risk scores for different actions, and the textual evidence for its decision.  Crucially, these risk numbers are cross-checked against the causal backbone's calculations by a guard script to prevent hallucination and ensure statistical validity. 

---

## 5. The "What-If" Workflow & Evaluation

The system is designed for interactive, counterfactual analysis.

* **Simulation:** A user can propose a hypothetical action (e.g., "switch to bupropion").  For the encoder, this action is added to the patient's record, which is re-embedded, and a new set of neighbors is analyzed with the AIPW estimator.  For the LLM, the instruction is simply prepended to its prompt. 
* **Evaluation:** The entire pipeline is subject to nightly regression tests, monitoring for model degradation, fairness, and calibration to ensure the system remains robust and reliable without constant human oversight. 

Multiple concurrent queries running on a server at once - langchain has asyncronous module - take one patient, submit all those queries (sequences) into the server and let it queue - vllm will tell you what the best number of submissions is for peak performance (e.g. 8-10 queries at a time)

Make sure ENTIRE patient is most similar to themsevles - throw in some changes in history and see how that changes things