# Project Methodology

## Objective

The goal is to create a clinical decision support system. For a given patient, this system will estimate and compare the probabilities of improvement for two scenarios: switching their current antidepressant medication versus staying on it. The final output is a risk report, such as “Switch → TRD 18 % (±4 %) vs. Stay 27 % (±5 %); ARR 9 %”.

## Methodology

The system is centered around a "population pool" composed of "visit sequences" from historical patients. This pool is used by two distinct methods to calculate risk.

### 1. Population Pool Construction

The visit sequences that form the pool are defined differently for two types of historical patients:

* **For "Switchers"**: A sequence represents a patient who made a medication switch in the past. The sequence starts with their first dose of the original medication, uses the date of the medication switch as its central "fulcrum point," and extends for a period after the switch to include the resulting outcome.
* **For "Stayers"**: A sequence represents a patient who did not switch medications. To form these sequences, a dynamic, query-dependent process is used. For each potential "stayer" patient, multiple visit sequences of varying lengths are generated. These sequences are then compared to the current patient being evaluated by the physician. The single sequence that is determined to be most similar to the current patient is the one selected for inclusion in the population pool.

### 2. Application Methods

Once the population pool is established, the risk is calculated via one of two paths:

#### Embedder Method

1.  From the population pool, the 50 nearest neighbor sequences are identified using an embedding-based search (FAISS).
2.  A pre-trained model is used to estimate the probability of each neighbor having switched or not.
    * **Model Type**: An XGBoost model with 3,000 trees and a depth of 6 is used.
    * **Model Training**: The model is trained on a feature matrix that includes a 512-dimensional encoder vector plus 12 named covariates for each patient. It is 5-fold cross-fitted and calibrated using the Platt method. The model estimates `π̂(x)=P(switch∣x)`, the probability of a switch given the patient's characteristics.
3.  An Inverse Probability Weighting (IPW) calculation is performed on this set of 50 neighbors to determine the final risk estimate.
    * **Weight Application**: The stabilized inverse-probability weights are pre-computed and stored. The 50 neighbors are joined with these weights.
    * **Final Calculation**: An Augmented Inverse Propensity Weighting (AIPW) estimator computes the local treatment effect (`τ̂_local`) in the set of weighted neighbors. This produces the final individualized risk percentages.

#### LLM Method

1.  From the population pool, the 1,000 nearest neighbor sequences are identified using an embedding-based search.
2.  These 1,000 sequences are provided to a Large Language Model (LLM).
3.  The LLM's task is to select a subset of the 50 "closest" sequences from the 1,000.
4.  The same IPW calculation, as detailed in the Embedder Method above, is then performed on this LLM-selected set of 50 neighbors to determine the final risk estimate.