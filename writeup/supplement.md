<!--
Supplementary material for the TRD prediction manuscript (writeup/manuscript.md).
Convert alongside the main text at submission: pandoc supplement.md -o supplement.docx.
-->

# Supplement S1. LLM clinical-similarity judge

## S1.1 Task

In the neighbor-weighted prediction arm, the LLM weighting strategy used a
large language model (`MedGemma-27B`) as a clinical-similarity judge. For each
anchor–neighbor pair the judge received the two patients' deterministic
narratives and returned a structured JSON score. Only the integer
`overall_similarity` field (0–100), rescaled to the unit interval, was used as
the neighbor weight; the per-dimension sub-scores and free-text lists were
retained for auditing only and did not enter the prediction. The judge was run
with cached results (each unordered pair scored once and stored), so repeated
runs are deterministic with respect to the cache.

## S1.2 System prompt (verbatim)

```
You are a clinical similarity scorer for major depressive disorder (MDD).
Compare two patient narratives that contain only structured EHR data from a fixed baseline window.

Judge similarity ONLY on factors that affect antidepressant response/tolerability.
Treat "Missing" as unknown (neutral). Treat "Absent" as truly absent. Do not infer beyond text.

Weight these dimensions when scoring (sum=100):
1. Baseline symptom phenotype (PHQ-9 subitems): 25 points
2. Psychiatric comorbidity & threat/anxiety/trauma & SUD/suicidality: 20 points
3. Medical/metabolic & pain/NSAIDs: 20 points
4. Treatment exposure & medication burden (polypharmacy, prior adequate trials, 3y complexity): 20 points
5. Social/functional access (no-show, marital/employment/SDOH): 10 points
6. Safety flags (contraindications): 5 points

Return JSON only.
Do not repeat items in lists.
Do NOT provide explanations, reasoning, or calculations. Output raw JSON only.
```

## S1.3 User prompt and output schema (verbatim)

The two narratives are injected at `{narrative_a}` and `{narrative_b}`.

```
INDEX PATIENT:
{narrative_a}

CANDIDATE PATIENT:
{narrative_b}

Limit "top_similarity_drivers" and "key_mismatches" to at most 5 items each. Return this JSON:
{
    "overall_similarity": 0-100,
    "phenotype": 0-100,
    "psych_comorbidity": 0-100,
    "metabolic_pain": 0-100,
    "treatment_burden": 0-100,
    "social_functional": 0-100,
    "safety": 0-100,
    "top_similarity_drivers": ["driver 1", "...up to 5"],
    "key_mismatches": ["mismatch 1", "...up to 5"]
}

Scoring rules:
- Same band (e.g., NSAID 0 vs 1) = small penalty; different band (0 vs >=2) = large penalty.
- Reward matches on polypharmacy, pain/NSAIDs, anxiety/fatigue most.
- If a dimension has 'Missing' data, strictly output the value 0 for that score.
```

## S1.4 Worked examples

The examples below are the extremes of the judged distribution across the 8B
encoder's neighborhoods: the five lowest- and five highest-scoring anchor–
neighbor pairs, extracted by `scripts/digital_twins/neighbors/llm_similarity_audit.py`
and provided in full at `test_data/llm_judge_low_{1..5}.txt` and
`test_data/llm_judge_high_{1..5}.txt`. Two of each are reproduced here in full;
the remaining responses are shown as JSON only.

### Lowest-similarity example (score 25) — dissimilar patients

> **Narrative A** — MDD (Recurrent, Severe); Female, 54, Married, smoker;
> Anxiety, SUD, PTSD present, suicidality flagged; thyroid + hyperlipidemia;
> 0 prior adequate AD trials; 2 active meds (citalopram, venlafaxine); 1 psych
> inpatient day; epilepsy and uncontrolled HTN present.
>
> **Narrative B** — MDD (Single Episode, Unspecified); Female, 76, Single,
> smoker; no psychiatric comorbidity, no suicidality; no medical comorbidity;
> 0 prior adequate AD trials; 0 active meds; no safety flags.

```json
{
    "overall_similarity": 25,
    "phenotype": 0, "psych_comorbidity": 25, "metabolic_pain": 0,
    "treatment_burden": 25, "social_functional": 0, "safety": 0,
    "top_similarity_drivers": [
        "Both patients are female smokers.",
        "Both patients have 0 prior adequate AD trials.",
        "Both patients have 0 benzodiazepine days (2y).",
        "Both patients have 0 psych inpatient days (2y).",
        "Both patients have 0 ED psych visits (2y)."
    ],
    "key_mismatches": [
        "Index patient has severe recurrent MDD vs candidate single episode.",
        "Index patient has anxiety, SUD, PTSD vs candidate has none.",
        "Index patient has epilepsy and uncontrolled HTN vs candidate has none."
    ]
}
```

### Highest-similarity example (score 98) — near-duplicate patients

> **Narrative A** — MDD (Single Episode, Unspecified); Female, 71, Divorced,
> never-smoker; Anxiety present, no SUD/suicidality; chronic pain +
> hyperlipidemia; 0 prior adequate AD trials; 1 active med (fluoxetine);
> uncontrolled HTN.
>
> **Narrative B** — Identical on every clinical field; differs only in age
> (67 vs 71) and recorded religion (Christian vs Baptist).

```json
{
    "overall_similarity": 98,
    "phenotype": 100, "psych_comorbidity": 100, "metabolic_pain": 100,
    "treatment_burden": 100, "social_functional": 100, "safety": 100,
    "top_similarity_drivers": [
        "Identical baseline symptom phenotype (PHQ-9 subitems)",
        "Identical psychiatric comorbidity (Anxiety present, SUD/Suicidality/Trauma absent)",
        "Identical medical comorbidity (Chronic Pain present, Diabetes/Thyroid absent)",
        "Identical treatment exposure (No prior adequate AD trials)",
        "Identical medication burden (1 active med, no NSAIDs)"
    ],
    "key_mismatches": [
        "Age difference (71 vs 67)",
        "Marital Status difference (Divorced vs Divorced)",
        "Religion difference (Baptist vs Christian)"
    ]
}
```

### Additional responses (JSON only)

Second-lowest pair (score 25):

```json
{
    "overall_similarity": 25,
    "phenotype": 0, "psych_comorbidity": 25, "metabolic_pain": 100,
    "treatment_burden": 0, "social_functional": 0, "safety": 100,
    "top_similarity_drivers": ["Absent NSAID burden", "Absent Chronic Pain", "Absent Epilepsy", "Absent Uncontrolled HTN", "Absent Diabetes"],
    "key_mismatches": ["Presence of Anxiety/SUD/PTSD/Adjustment Disorder in Index", "Active polypharmacy (2 meds) in Index", "Different MDD severity/episode type"]
}
```

Second-highest pair (score 98):

```json
{
    "overall_similarity": 98,
    "phenotype": 0, "psych_comorbidity": 100, "metabolic_pain": 100,
    "treatment_burden": 100, "social_functional": 100, "safety": 100,
    "top_similarity_drivers": ["No psychiatric comorbidity", "No substance use disorder", "No chronic pain", "No NSAID burden", "No prior adequate antidepressant trials"],
    "key_mismatches": ["Age difference (76 vs 60)", "Missing BMI", "Missing BP"]
}
```

## S1.5 Interpretation and caveats

The judge's **overall** similarity behaved sensibly at the extremes — near-
duplicate patients scored 98, clinically dissimilar patients scored 25 — and
the population of ~523,000 cached judgements spans roughly 35–98 (modal
45–65), so the score is graded rather than degenerate.

The **per-dimension sub-scores and free-text rationales are not reliable** and
should be read as illustrative only:

- The `overall_similarity` is a holistic judgment, **not an aggregation of the
  six sub-scores**: the second-lowest pair scores `metabolic_pain` 100 and
  `safety` 100 yet an `overall` of 25, and the second-highest pair scores
  `phenotype` 0 yet an `overall` of 98.
- The judge **confabulates rationale**: the highest pair cites "identical
  PHQ-9 subitems" although no PHQ-9 subitems appear in the narrative, and lists
  "Divorced vs Divorced" — two identical values — as a key mismatch. The
  second-highest pair lists "Missing BMI/BP" as mismatches despite the system
  prompt instructing that Missing be treated as neutral.

Because only `overall_similarity` enters the neighbor weighting, these
sub-score artifacts do not propagate into the predictions; they are reported
here for transparency. Consistent with the main-text result, the weighting
strategy was second-order to the retrieval scheme: where the nearest neighbors
were already maximally similar, LLM weighting added no discrimination over
cosine weighting (both ROC AUC ≈ 0.624), and it lifted discrimination only
under random or subsampled retrieval, where the overall similarity score could
recover the few congruent neighbors a non-targeted draw happened to include —
behavior attributable to the retrieval geometry rather than to any deficiency
in the overall similarity score.

# Supplement S2. Effective dimensionality of the embedded representation

Individual embedding dimensions carry no clinical meaning, so per-feature
interpretability — available for the rule-based vector in main-text Figure 4 —
does not transfer to the 4,096-dimensional embedded representation. Instead we
characterize *how many* latent dimensions carry the predictive signal, using
three complementary views. All figures are for the primary
`Qwen3-Embedding-8B` encoder on the held-out test set.

## S2.1 Sparsity and cumulative built-in importance

The best logistic-regression fit retained all 4,096 embedding dimensions with
nonzero coefficients (no L1 sparsity: 4,096 of 4,096 coefficients nonzero), so
the signal is not concentrated in a small subset of dimensions. Ranking
dimensions by each fitted classifier's native importance (`|coef|` for logistic
regression, `feature_importances_` for the tree ensembles) and accumulating the
importance mass confirms a diffuse distribution: for logistic regression, 80%
of the coefficient magnitude is spread across roughly 2,067 of the 4,096
dimensions and 90% across roughly 2,659 (Figure S1). TRD-relevant information
is therefore distributed broadly across the embedding rather than localized.

![](../results/Qwen-Qwen3-Embedding-8B/google_medgemma-27b-text-it/feature_importance/feature_importance_cumulative_EMBEDDED.png){width=80%}

***Figure S1.** Cumulative built-in feature importance for the four embedded
classifiers. Each curve plots the cumulative fraction of total importance mass
against dimension rank (dimensions sorted by descending native importance); the
K₈₀ / K₉₀ knees are reported in the legend. A diagonal-like curve indicates a
diffuse, high-effective-rank signal.*

## S2.2 Cumulative univariate correlation

An importance ranking is model-specific. As a model-agnostic check we ranked
dimensions by the absolute Spearman correlation between each dimension and the
outcome, |ρ(dim, y)|, and overlaid that baseline against per-classifier curves
ranking dimensions by |ρ(dim, risk score)| (Figure S2). Divergence between the
model-agnostic baseline and a classifier's curve would flag dimensions the
classifier weights through regularization or interactions that a univariate
ranking cannot see; the curves were broadly concordant and similarly diffuse,
consistent with the approximately linear, high-effective-rank structure. Only
|ρ| is used — the sign of a correlation on an unnamed latent dimension is not
interpretable.

![](../results/Qwen-Qwen3-Embedding-8B/google_medgemma-27b-text-it/feature_importance/feature_correlation_cumulative_EMBEDDED.png){width=80%}

***Figure S2.** Cumulative absolute univariate (Spearman) correlation. One
model-agnostic baseline curve ranks dimensions by |ρ(dim, outcome)|; the four
per-classifier curves rank by |ρ(dim, predicted risk)|. Cumulative fraction of
total |ρ| mass versus rank, with K₈₀ / K₉₀ knees in the legend.*

## S2.3 PCA-K discrimination sweep

To locate the geometric plateau, each classifier was retrained on the top
K ∈ {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024} principal components of the
embedding and its held-out ROC AUC plotted against K (Figure S3).
Discrimination rose steeply over the first handful of components and then
plateaued, with no single low-dimensional projection recovering the full-rank
performance — the same broadly-distributed-signal conclusion reached from the
cumulative-importance and correlation curves.

![](../results/Qwen-Qwen3-Embedding-8B/google_medgemma-27b-text-it/feature_importance/feature_importance_pca_sweep_logistic_regression_EMBEDDED.png){width=80%}
![](../results/Qwen-Qwen3-Embedding-8B/google_medgemma-27b-text-it/feature_importance/feature_importance_pca_sweep_random_forest_EMBEDDED.png){width=80%}
![](../results/Qwen-Qwen3-Embedding-8B/google_medgemma-27b-text-it/feature_importance/feature_importance_pca_sweep_gradient_boosting_EMBEDDED.png){width=80%}
![](../results/Qwen-Qwen3-Embedding-8B/google_medgemma-27b-text-it/feature_importance/feature_importance_pca_sweep_xgboost_EMBEDDED.png){width=80%}

***Figure S3.** ROC AUC versus number of retained principal components, one
panel per classifier: (A) logistic regression, (B) random forest,
(C) gradient boosting, (D) XGBoost. The plateau marks the effective number of
principal directions beyond which added components do not improve
discrimination.*

# Supplement S3. Precision–recall performance

The main text reports discrimination as ROC AUC. Under the cohort's ~8:1 class
imbalance (10.6% TRD-positive), ROC AUC can flatter apparent performance
because the true-negative-dominated specificity term stays high regardless of
how the positive class is ranked. The area under the precision–recall curve
(AUPRC) is the complementary view: its no-skill baseline is the positive rate
itself (0.106), not 0.5, and it exposes the precision cost of capturing TRD
cases. We report it here rather than in the main text.

AUPRC tracked ROC AUC closely and remained modest for every model (Table S1),
consistent with a representation suited to triage and risk-stratification
rather than standalone rule-in: at this base rate, capturing high TRD recall
necessarily forces low precision.

***Table S1.** AUPRC of the four classifiers on each representation (held-out
test set). No-skill baseline = 0.106 (the positive rate).*

| Representation | Classifier | AUPRC |
| --- | --- | ---: |
| EMBEDDED | Logistic regression | 0.248 |
| EMBEDDED | Random forest | 0.219 |
| EMBEDDED | Gradient boosting | 0.230 |
| EMBEDDED | XGBoost | 0.232 |
| FEATURE | Logistic regression | 0.235 |
| FEATURE | Random forest | 0.248 |
| FEATURE | Gradient boosting | 0.228 |
| FEATURE | XGBoost | 0.242 |

***Table S2.** Embedded logistic-regression AUPRC by encoder (held-out test
set). No-skill baseline = 0.106.*

| Encoder | AUPRC |
| --- | ---: |
| bge-small-en-v1.5 | 0.204 |
| bge-en-icl | 0.232 |
| Qwen3-Embedding-4B | 0.236 |
| Qwen3-Embedding-8B | 0.248 |

![](../results/Qwen-Qwen3-Embedding-8B/google_medgemma-27b-text-it/pr_curves/pr_curve_logistic_regression_EMBEDDED.png){width=80%}
![](../results/Qwen-Qwen3-Embedding-8B/google_medgemma-27b-text-it/pr_curves/pr_curve_random_forest_FEATURE.png){width=80%}

***Figure S4.** Precision–recall curves for the best classifier on each
representation (held-out test set; primary `Qwen3-Embedding-8B` encoder).
(A) Embedded logistic regression; (B) rule-based random forest. The horizontal
reference is the no-skill baseline (0.106, the positive rate).*

# Supplement S4. Calibration absolute-error metrics

The main text reports calibration shape via the calibration slope and intercept
(Table 5) and the calibration curves (Figure 3). Here we report the two
absolute-error summaries: the Brier score and a weighted calibration error
(WCE), both lower-is-better (Table S3). Brier scores were uniformly near
0.089–0.091 across all eight models, dominated by the low base rate (10.6%
positive); the WCE separated the models more, with random forest on the
embedded representation and logistic regression on the feature vector the
lowest (0.006 each), consistent with their near-ideal slopes in Table 5.

***Table S3.** Brier score and weighted calibration error (WCE) of the four
classifiers on each representation (held-out test set). Lower is better for
both. Calibration slope and intercept are in main-text Table 5.*

| Representation | Classifier | Brier | WCE |
| --- | --- | ---: | ---: |
| EMBEDDED | Logistic regression | 0.090 | 0.020 |
| EMBEDDED | Random forest | 0.091 | 0.006 |
| EMBEDDED | Gradient boosting | 0.091 | 0.007 |
| EMBEDDED | XGBoost | 0.090 | 0.015 |
| FEATURE | Logistic regression | 0.090 | 0.006 |
| FEATURE | Random forest | 0.089 | 0.014 |
| FEATURE | Gradient boosting | 0.090 | 0.019 |
| FEATURE | XGBoost | 0.090 | 0.020 |
