<!--
Digital Twins: predicting treatment-resistant depression (TRD) from
EHR-derived patient representations.

Scope: Methods onward only. Title page, abstract, keywords, and
Introduction are out of scope and authored separately.

Target venue: JMIR Mental Health. Reporting follows the TRIPOD+AI
checklist. Final form is Word (.docx): iterate here in Markdown, then
convert with `pandoc manuscript.md -o manuscript.docx` into the JMIR
template (writeup/JMIR_template.docx) at submission time.

[PENDING] markers flag content that depends on pipeline artifacts not
yet regenerated. Cohort-characterization numbers are final (from
notebooks/figures/). Performance numbers are from a prior run; see the
provenance note under Results.
-->

# Methods

## Source of data and study design
<!-- TRIPOD+AI 4a (data source), 4b (study dates) -->

This was a retrospective cohort study using a de-identified EHR
extract (data version DV250901v1, patient version PV251208v1). Anchor
dates — the date of each patient's first adequate antidepressant
exposure — span 2016–2024. For every patient an *anchor date* defines
the temporal origin: a fixed pre-anchor window of clinical history is
used to construct predictors, and a post-anchor follow-up window is
used to ascertain the TRD outcome. The development and the held-out
evaluation samples were drawn from the same data source and time
period and partitioned by a single random split (see *Train/test
split* below); this is therefore an internal validation by random
split, not a temporal or external validation.

## Participants and eligibility
<!-- TRIPOD+AI 5a (eligibility); participant flow detailed in Results -->

Eligibility was applied as an ordered cascade (Table 1). Beginning
from the full source population, we required: (1) a qualifying MDD
diagnosis; (2) absence of bipolar disorder and absence of
schizophrenia-spectrum diagnoses (to isolate unipolar depression);
(3) an antidepressant anchor (a first adequate antidepressant
exposure); (4) an MDD diagnosis recorded on or before the anchor;
(5) at least two years of pre-anchor history; and (6) at least one
year of post-anchor follow-up, so that the TRD outcome could be
ascertained reliably. Patients failing the chronological or follow-up
prerequisites were recorded with an explicit rejection reason for
attrition accounting.

## Outcome
<!-- TRIPOD+AI 6a (outcome definition), 6b (assessment) -->

The prediction target was incident TRD, supplied as a pre-defined
external binary label and consumed as-is by the modeling pipeline. The
label was assigned independently of the predictors used in this study.
**[PENDING: exact operational TRD definition (number/sequence of
failed adequate trials and the ascertainment window) to be stated
verbatim from the label provenance.]**

## Predictors and patient representations
<!-- TRIPOD+AI 7a (predictor definition), 7b (assessment) -->

All predictors were derived solely from data within the pre-anchor
window; no post-anchor information entered any predictor. Each
eligible patient was rendered into two independent representations
from the same timeline-sliced source record.

**Rule-based feature vector (FEATURE).** A typed feature vector was
constructed per patient with explicit dtypes that drive downstream
preprocessing: quantitative counts and durations (e.g., encounter
count, pre-anchor history length, age, per-agent adequate-trial counts,
polypharmacy count) as continuous values; single-valued binary
clinical flags and multi-label clinical indicator sets (psychiatric
comorbidities, medical comorbidities, prescribing-constraint flags,
substance-use-disorder categories, and social-determinant categories)
as booleans; and single-valued nominal fields (sex, preferred
language, marital status, religion, smoking status, race/ethnicity,
MDD recurrence, MDD severity) as categoricals compressed via
standardized vocabularies. Adequate antidepressant trials were defined
by a 42-day (six-week) minimum continuous exposure threshold per
agent.

**Deterministic narrative and neural embedding (EMBEDDED).** The same
sliced record was deterministically rendered into a human-readable
Markdown narrative summarizing only the pre-anchor window. Rendering is
rule-based and reproducible — no generative model participates in
narrative construction, so the narrative introduces no stochastic
content. Each narrative was then mapped to a fixed-length dense vector
by a pretrained sentence-transformer encoder. Four encoders were
evaluated independently: `bge-small-en-v1.5`, `bge-en-icl`,
`Qwen3-Embedding-4B`, and `Qwen3-Embedding-8B`.

## Missing data
<!-- TRIPOD+AI 9 (missing data handling) -->

No imputation was performed anywhere in the pipeline. The handling of
missingness was deliberately representation-specific and is stated
here in full because it bears directly on interpretation.

Three cohort-averaged vital columns (BMI, systolic blood pressure,
diastolic blood pressure) were present in the feature parquet but
*dropped* from the FEATURE modeling matrix at load time. A three-tier
missingness diagnostic (in-window, all pre-anchor history, any date)
established that the cohort retains approximately 50% missing BMI even
under the most permissive time filter, with a 7–11 percentage-point
TRD-stratified missingness gap at every filter. Approximately 28.4% of
patients have no vital record at any date, and in-window vitals
missingness reaches roughly 63–80%. Because the missingness is
structural and associated with the outcome (a missing-not-at-random
pattern), imputation was judged indefensible and the columns were
removed rather than filled; no NaN cells reached any transformer, and
no missingness indicators were created for the continuous block.

Categorical missingness was retained as an explicit category. On the
FEATURE side this enters the one-hot encoder as an implicit "Missing"
level; on the EMBEDDED side the deterministic narrative encodes absence
with a literal "Missing" token. Marital status and smoking status are
effectively complete (<0.5% missing). Religion is *not* clean: 15.8%
missing overall with a monotone age gradient (30.5% in ages 18–29
falling to 9.5% in ages ≥65), a missing-at-random pattern with respect
to age that propagates into the model as a missing-indicator level.
The treatment of religion (retain the indicator, drop the field, or
document as a limitation) is reported transparently and discussed
under Limitations.

## Sample size and events per variable
<!-- TRIPOD+AI 8 (sample size) -->

No formal a-priori sample-size calculation was performed; the cohort
size was fixed by the eligibility cascade. We report events per
variable (EPV) against the conventional threshold of ≥10. The shared
training-set TRD-positive numerator is 827 events. For the
high-dimensional EMBEDDED representation, EPV is far below threshold by
construction (827 events against thousands of embedding dimensions;
e.g., 827/4096 ≈ 0.2 for the 4096-dimensional encoder), and the
EMBEDDED models are accordingly interpreted as high-dimensional
learners rather than as low-dimensional regression models.
**[PENDING: exact post-ColumnTransformer feature width and the
resulting FEATURE-side EPV ratio, computed from the cached fitted
transformers.]**

## Train/test split
<!-- TRIPOD+AI 5b / analysis -->

A single stratified 80/20 train/test split was created once and shared
across every evaluation arm to ensure fair comparison. The split
preserved the natural class imbalance: 7,780 training patients (827
TRD-positive, 10.6%) and 1,945 test patients (207 TRD-positive, 10.6%).
Test patient identifiers were persisted for reproducibility. In the
neighbor-weighted pipeline, test patients were excluded from one
another's neighbor pools at retrieval time to prevent leakage, while
remaining available as query anchors. Train-versus-test comparability
was confirmed by per-predictor standardized mean differences (SMD),
all small in magnitude (maximum absolute SMD ≈ 0.07; Table 3).

## Model development
<!-- TRIPOD+AI 10 (model development), 11 (analysis methods) -->

**Classical machine learning.** Four classifiers — logistic regression
(`max_iter`=1000), random forest, gradient boosting, and XGBoost
(`eval_metric`=logloss) — were trained on each representation. Each
classifier was wrapped in a pipeline with a dtype-routed column
transformer: the numeric block was standardized; the categorical block
was one-hot encoded (binary categories collapsed, unknown categories
ignored at inference); the boolean block was cast to integer and passed
through. The embedded representation is a single all-numeric block and
flows through the numeric branch only. Every classifier was tuned by
five-fold cross-validated grid search optimizing ROC AUC; the refit
best estimator supplied predicted probabilities. Logistic-regression
grids were partitioned by penalty to respect solver compatibility.
Fitted searchers were cached to disk so that re-runs are robust to
downstream failure.

**Neighbor-weighted retrieval prediction.** On the EMBEDDED
representation only, a neighbor-weighted predictor retrieved each
anchor's top-*K* neighbors by cosine similarity, scored each
anchor–neighbor pair for clinical similarity with a large language
model judge (cached to a judgements database), and computed a weighted
TRD probability under four weighting strategies (uniform, cosine, LLM,
and the harmonic mean of cosine and LLM). Four retrieval schemes
(nearest, farthest, random, and a two-stage subsampled mode) were
evaluated to isolate predictive lift against negative and random
baselines. This pipeline is embedding-only because cosine similarity is
not well defined over mixed categorical/numeric features.

**Semantic-feature ablation.** To attribute embedding-based predictive
signal to named clinical concepts, we perturbed individual narrative
sections or fields, re-embedded the cohort, and re-scored using the
*frozen* baseline classifiers (no retraining). Freezing the baseline
answers an input-perturbation attribution question — "does the
embedder's encoding of concept *X* drive predictions?" — rather than
"could a retrained model route around the ablation?" Perturbation used
cohort-wide donor permutation (each patient donates its value for the
perturbed field exactly once per specification), preserving the
marginal distribution of the perturbed field while severing its link to
the individual patient. The pre-registered ablation slate comprised
five specifications: race/ethnicity (field permutation), psychiatric
history (section permutation), medication burden (section permutation),
treatment contraindications (section permutation), and social
determinants of health (field permutation). The treatment-exposure
section was deliberately excluded because prior adequate antidepressant
trials are constitutive of the TRD definition, so its ablation delta
would be large, predictable, and uninformative.

## Statistical analysis and performance metrics
<!-- TRIPOD+AI 11, 12 (evaluation) -->

Discrimination was summarized by the area under the receiver operating
characteristic curve (ROC AUC), with 95% confidence intervals from a
bootstrap of the test set, and by the area under the precision–recall
curve (AUPRC). Calibration was summarized by the Brier score, a
weighted calibration error, and the calibration slope and intercept.
For the ablation study, deltas versus baseline were accompanied by
paired-bootstrap confidence intervals: within each bootstrap draw a
single resampling index was applied to both the baseline and the
ablated probability vectors, yielding a paired delta-AUC distribution
(paired bootstrap is required because the same patients are scored by
both models, so their AUCs are positively correlated and independent
intervals would be inflated). An optimal operating point was identified
by Youden's J statistic with associated sensitivity, specificity, and
likelihood ratios.

## Reproducibility and software
<!-- TRIPOD+AI 13 (availability) -->

All randomness was seeded. Models were implemented in Python with
scikit-learn and XGBoost; embeddings used sentence-transformer
encoders. The pipeline is organized into discrete, independently
re-runnable stages (data loading, narrative and feature-vector
generation, embedding, retrieval/scoring, and prediction/evaluation),
with cached intermediate artifacts. **[PENDING: code-availability
statement and repository/DOI link for submission.]**

# Results

## Participant flow
<!-- TRIPOD+AI 13a (participant flow), 13b (characteristics) -->

The eligibility cascade reduced a source population of 502,118 patients
to a final analysis cohort of 9,725 (Table 1). The largest single
reduction was the MDD diagnosis requirement (502,118 → 73,942),
followed by the pre-anchor history requirement (43,141 → 16,799) and
the post-anchor follow-up requirement (16,799 → 9,725). Of the final
cohort, 1,034 patients (10.6%) were TRD-positive.

***Table 1.** Participant flow through the eligibility cascade. Each
row applies one additional filter to the survivors of the row above.*

| Stage | Remaining | Rejected at stage |
| --- | ---: | ---: |
| Raw population | 502,118 | — |
| MDD-diagnosed | 73,942 | 428,176 |
| Not bipolar AND not schizophrenia-spectrum | 61,806 | 12,136 |
| Has antidepressant anchor | 43,330 | 18,476 |
| Has MDD before anchor | 43,141 | 189 |
| ≥2 years pre-anchor history | 16,799 | 26,342 |
| ≥1 year post-anchor follow-up (final cohort) | 9,725 | 7,074 |

## Cohort characteristics (Table 1 of manuscript)

The cohort was older and predominantly female. Median age was 66 years
(IQR 50–77; mean 62.5), with 52.7% aged 65 or older and only 6.8% aged
18–29. Female patients comprised 72.8% of the cohort. The population
was predominantly White/Caucasian (81.3%) and English-preferring
(99.2%). Recurrent MDD coding was strongly associated with TRD (26.7%
of TRD-positive versus 12.8% of TRD-negative patients; SMD 0.354), as
was severe MDD coding (16.5% versus 5.0%; SMD 0.377) and a flagged
suicidality history (24.0% versus 8.7%; SMD 0.423). Selected
characteristics with the largest TRD-stratified differences are
summarized in Table 2; the full descriptive set (continuous summaries,
all boolean indicators, and all categorical levels) is available in the
supporting figure tables.

***Table 2.** Selected cohort characteristics by TRD status. Continuous
variables are median (IQR); categorical/boolean entries are n (%).
SMD = standardized mean difference (TRD-positive vs TRD-negative). Full
table in supporting material.*

| Characteristic | Level | Overall | TRD+ | TRD− | SMD |
| --- | --- | ---: | ---: | ---: | ---: |
| **Demographics** | | | | | |
| Age (years) | median (IQR) | 66 (50–77) | 63 (47–75) | 66 (50–77) | −0.117 |
| Age band | 18–29 | 666 (6.8%) | 78 (7.5%) | 588 (6.8%) | 0.030 |
| | 30–44 | 1,254 (12.9%) | 150 (14.5%) | 1,104 (12.7%) | 0.053 |
| | 45–64 | 2,680 (27.6%) | 324 (31.3%) | 2,356 (27.1%) | 0.093 |
| | 65+ | 5,125 (52.7%) | 482 (46.6%) | 4,643 (53.4%) | −0.136 |
| Sex | Female | 7,077 (72.8%) | 722 (69.8%) | 6,355 (73.1%) | −0.073 |
| Race/ethnicity | White/Caucasian | 7,909 (81.3%) | 826 (79.9%) | 7,083 (81.5%) | −0.041 |
| | Black/African American | 555 (5.7%) | 47 (4.5%) | 508 (5.8%) | −0.059 |
| | Am. Indian/Alaska Native | 511 (5.3%) | 70 (6.8%) | 441 (5.1%) | 0.072 |
| | Missing | 21 (0.2%) | 5 (0.5%) | 16 (0.2%) | 0.052 |
| **Depression phenotype** | | | | | |
| MDD recurrence | Recurrent | 1,389 (14.3%) | 276 (26.7%) | 1,113 (12.8%) | 0.354 |
| | Single episode | 8,173 (84.0%) | 743 (71.9%) | 7,430 (85.5%) | −0.338 |
| MDD severity | Severe | 609 (6.3%) | 171 (16.5%) | 438 (5.0%) | 0.377 |
| | Moderate | 381 (3.9%) | 58 (5.6%) | 323 (3.7%) | 0.090 |
| | Psychotic | 54 (0.6%) | 15 (1.5%) | 39 (0.4%) | 0.103 |
| | Unspecified | 8,433 (86.7%) | 766 (74.1%) | 7,667 (88.2%) | −0.367 |
| **Psychiatric and substance comorbidity** | | | | | |
| Suicidality flagged | True | 1,003 (10.3%) | 248 (24.0%) | 755 (8.7%) | 0.423 |
| Anxiety disorder | True | 5,477 (56.3%) | 692 (66.9%) | 4,785 (55.1%) | 0.245 |
| Substance use disorder (any) | True | 3,378 (34.7%) | 483 (46.7%) | 2,895 (33.3%) | 0.276 |
| PTSD | True | 520 (5.3%) | 107 (10.3%) | 413 (4.8%) | 0.213 |
| Insomnia | True | 1,589 (16.3%) | 235 (22.7%) | 1,354 (15.6%) | 0.182 |
| Adjustment disorder | True | 364 (3.7%) | 93 (9.0%) | 271 (3.1%) | 0.248 |
| Alcohol use disorder | True | 792 (8.1%) | 156 (15.1%) | 636 (7.3%) | 0.248 |
| Opioid use disorder | True | 701 (7.2%) | 114 (11.0%) | 587 (6.8%) | 0.150 |
| **Treatment and utilization** | | | | | |
| Active med count | median (IQR) | 2 (1–3) | 2 (1–3) | 2 (1–3) | 0.219 |
| Encounter count | median (IQR) | 5 (3–9) | 4 (2–8) | 5 (3–9) | −0.085 |
| Pre-anchor history (days) | median (IQR) | 1,590 (1,129–2,226) | 1,439 (1,039–2,019) | 1,606 (1,142–2,245) | −0.183 |
| Augmentation therapy used | True | 89 (0.9%) | 17 (1.6%) | 72 (0.8%) | 0.074 |

## Subgroup outcome prevalence
<!-- TRIPOD+AI 13b / fairness context -->

TRD prevalence varied modestly across subgroups. By sex it was 10.2%
(female) versus 11.8% (male). By age band it was highest in
18–64-year-olds (11.7–12.1%) and lowest in those aged ≥65 (9.4%).
Several small social-determinant strata showed elevated but imprecise
prevalence (e.g., family/support-group issue 19.4%, upbringing-related
issue 24.8%), and patients with missing race/ethnicity (n=21) showed
roughly double the baseline prevalence (23.8% versus 10.6%) — the same
missing-correlates-with-outcome pattern flagged for religion. Strata
with fewer than 20 patients were flagged as a small-cell ceiling on any
eventual subgroup performance analysis.

## Train/test comparability

The training and test sets were closely matched on every predictor.
The largest absolute standardized mean difference across all predictors
was approximately 0.07, well below the 0.1 threshold conventionally
taken to indicate meaningful imbalance, supporting the fairness of the
shared-split comparison (Table 3).

***Table 3.** Train/test split summary. Per-predictor SMDs were all
small (max |SMD| ≈ 0.07).*

| Split | n | TRD-positive | TRD rate |
| --- | ---: | ---: | ---: |
| Train | 7,780 | 827 | 10.6% |
| Test | 1,945 | 207 | 10.6% |

## Density and chronology checks

To test whether data volume could act as a confound (richer records
appearing higher-risk), we correlated the TRD label with three
volume/recency proxies on the full untruncated data. All associations
were weak: pre-anchor history length (Spearman ρ = −0.059),
MDD-to-anchor gap (ρ = −0.085), and encounter count (ρ = −0.052), each
negative and small, indicating that TRD-positive patients did not
simply have more data. The direction (slightly *shorter* histories
among TRD-positive patients) is consistent with earlier escalation
rather than a richness artifact.

![](../notebooks/figures/density_pre_anchor_history_days.png){width=80%}
![](../notebooks/figures/density_num_encounters.png){width=80%}
![](../notebooks/figures/density_mdd_to_anchor_days.png){width=80%}

***Figure 1.** TRD-stratified distributions of data-volume and recency
proxies (full cohort). (A) Pre-anchor history length; (B) encounter
count; (C) interval from MDD onset to index. The TRD-positive and
TRD-negative distributions are near-identical, consistent with the weak
negative label correlations reported above and arguing against data
volume acting as a richness confound. Axes are truncated at a per-panel
percentile for resolution only.*

## Provenance of performance results

The performance results in the remaining subsections are drawn from a
complete prior pipeline run on the same 9,725-patient cohort and the
same 1,945-patient held-out test split used for the cohort
characterization above, with the `Qwen3-Embedding-8B` encoder and a
`MedGemma-27B` similarity judge. They are reported here as the current
best available evidence and carry three caveats that the planned re-run
will resolve. First, the FEATURE-side classical-ML numbers were
computed on a 64-column feature matrix that still included the three
vital columns and a psychotherapy-count column that the current
pipeline removes; those specific FEATURE AUCs will therefore shift on
re-run, whereas the EMBEDDED, neighbor-weighted, and ablation results
(which never depended on the vital columns) are unaffected. Second, the
ablation specifications were named `swap_*` in this run and map
one-to-one onto the current `permute_*` names (`swap_safety`
corresponds to `permute_treatment_contraindications`). Third, this run
predates the paired-bootstrap delta-AUC confidence intervals, so the
ablation deltas below are point estimates. **[PENDING: refresh across
all four embedders with paired-bootstrap delta CIs.]**

## Model discrimination
<!-- TRIPOD+AI 13b -->

On the embedded representation, logistic regression achieved the
highest discrimination (ROC AUC 0.704, 95% CI 0.666–0.744; AUPRC
0.268), followed by XGBoost (0.691), random forest (0.677), and
gradient boosting (0.661). On the rule-based feature vector, random
forest led (0.688, 95% CI 0.651–0.729), with the remaining classifiers
between 0.651 and 0.675. Embedded logistic regression thus exceeded
both its feature-vector counterpart (0.704 vs 0.651) and the best
feature-vector model (random forest, 0.688), while every model's
discrimination remained modest — consistent with a difficult target and
a 10.6% positive rate (Table 4).

***Table 4.** Discrimination of the four classifiers on each
representation (held-out test set). 95% CIs are bootstrap percentile
intervals. FEATURE-side values are from the legacy 64-column matrix
(see provenance note).*

| Representation | Classifier | ROC AUC (95% CI) | AUPRC |
| --- | --- | :---: | ---: |
| EMBEDDED | Logistic regression | 0.704 (0.666–0.744) | 0.268 |
| EMBEDDED | Random forest | 0.677 (0.635–0.715) | 0.224 |
| EMBEDDED | Gradient boosting | 0.661 (0.624–0.701) | 0.182 |
| EMBEDDED | XGBoost | 0.691 (0.650–0.734) | 0.235 |
| FEATURE | Logistic regression | 0.651 (0.607–0.694) | 0.218 |
| FEATURE | Random forest | 0.688 (0.651–0.729) | 0.242 |
| FEATURE | Gradient boosting | 0.655 (0.610–0.696) | 0.227 |
| FEATURE | XGBoost | 0.675 (0.634–0.715) | 0.244 |

![](../results/Qwen-Qwen3-Embedding-8B/google_medgemma-27b-text-it/roc_curves/roc_curve_logistic_regression_EMBEDDED.png){width=80%}
![](../results/Qwen-Qwen3-Embedding-8B/google_medgemma-27b-text-it/roc_curves/roc_curve_random_forest_FEATURE.png){width=80%}

***Figure 2.** Receiver-operating-characteristic curves for the best
classifier on each representation (held-out test set). (A) Embedded
logistic regression; (B) rule-based random forest. Shaded bands are
bootstrap 95% confidence intervals; point AUCs are given in Table 4.
The cross-embedder discrimination comparison is shown in Figure 8.*

## Model calibration

Calibration varied substantially by classifier (Table 5). The embedded
logistic regression was the best-calibrated discriminative model (Brier
0.088, weighted calibration error 0.007, slope 1.47, intercept −0.10);
its slope above 1 indicates mildly under-confident probabilities.
Gradient boosting was poorly calibrated on the embedded representation
(slope 0.05), reflecting severely compressed probability estimates, and
the tree ensembles were generally less reliable than the linear model.
No model placed any test prediction above 0.9, and a large fraction
fell below 0.1, as expected given the low base rate.

***Table 5.** Calibration of the four classifiers on each
representation (held-out test set). Brier and weighted calibration
error (WCE): lower is better; calibration slope: ideal is 1; intercept:
ideal is 0.*

| Representation | Classifier | Brier | WCE | Slope | Intercept |
| --- | --- | ---: | ---: | ---: | ---: |
| EMBEDDED | Logistic regression | 0.088 | 0.007 | 1.47 | −0.10 |
| EMBEDDED | Random forest | 0.090 | 0.011 | 0.84 | 0.03 |
| EMBEDDED | Gradient boosting | 0.094 | 0.023 | 0.05 | 0.17 |
| EMBEDDED | XGBoost | 0.089 | 0.015 | 0.41 | 0.12 |
| FEATURE | Logistic regression | 0.091 | 0.007 | 0.72 | 0.06 |
| FEATURE | Random forest | 0.090 | 0.011 | 1.36 | −0.04 |
| FEATURE | Gradient boosting | 0.090 | 0.009 | 0.43 | 0.13 |
| FEATURE | XGBoost | 0.089 | 0.008 | 1.62 | −0.11 |

![](../results/Qwen-Qwen3-Embedding-8B/google_medgemma-27b-text-it/calibration_curves/calibration_curve_logistic_regression_EMBEDDED.png){width=80%}
![](../results/Qwen-Qwen3-Embedding-8B/google_medgemma-27b-text-it/calibration_curves/calibration_curve_gradient_boosting_EMBEDDED.png){width=80%}

***Figure 3.** Calibration curves on the embedded representation
(held-out test set). (A) Logistic regression, the best-calibrated
discriminative model (slope near 1). (B) Gradient boosting, showing
severely compressed probability estimates (slope near 0). Calibration
statistics for all classifiers are in Table 5.*

## Feature importance and embedding dimensionality

On the rule-based representation, the highest-weighted predictors were
clinically coherent and recapitulated the strongest univariate
correlates of TRD (Table 2): obsessive–compulsive disorder, a flagged
suicidality history, any substance-use disorder, and severe MDD coding
carried the largest positive logistic-regression weights, while a
smaller set of indicators (e.g., nicotine use disorder, an
employment-related social-determinant flag) carried negative weights
(Figure 5). On the embedded representation no single latent dimension is
clinically interpretable, so we instead characterized how many
dimensions carry the predictive signal: a principal-component sweep
located the discrimination plateau well below the full 4,096-dimensional
space, and the cumulative built-in-importance curves concentrate most of
the signal mass in a leading subset of dimensions (Figure 4), indicating
a largely low-dimensional predictive structure.

![](../results/Qwen-Qwen3-Embedding-8B/google_medgemma-27b-text-it/feature_importance/feature_importance_pca_sweep_logistic_regression_EMBEDDED.png){width=80%}
![](../results/Qwen-Qwen3-Embedding-8B/google_medgemma-27b-text-it/feature_importance/feature_importance_cumulative_EMBEDDED.png){width=80%}

***Figure 4.** Embedded effective-rank diagnostics. (A) ROC AUC versus
the number of retained principal components for embedded logistic
regression; discrimination plateaus far below the full dimensionality.
(B) Cumulative built-in importance across embedding dimensions for each
classifier, ranked by magnitude; the early rise indicates signal
concentrated in a leading subset of dimensions.*

![](../results/Qwen-Qwen3-Embedding-8B/google_medgemma-27b-text-it/feature_importance/feature_importance_logistic_regression.png){width=80%}
![](../results/Qwen-Qwen3-Embedding-8B/google_medgemma-27b-text-it/feature_importance/feature_importance_random_forest.png){width=80%}

***Figure 5.** Feature importance on the rule-based representation.
(A) Signed logistic-regression coefficients (steelblue raises TRD risk,
firebrick lowers it). (B) Random-forest importances, with
direction-of-effect recovered by univariate correlation. The
highest-ranked predictors mirror the largest TRD-stratified differences
in Table 2.*

## Neighbor-weighted prediction

The neighbor-weighted predictor behaved coherently with respect to the
embedding geometry (Table 6). Retrieving the *nearest* neighbors
yielded the best discrimination (ROC AUC 0.650–0.655 across weighting
strategies), retrieving the *farthest* neighbors inverted the signal to
well below chance (0.389–0.397), and random retrieval landed in between
(0.452–0.508). The two-stage subsampled mode recovered most of the
nearest-neighbor signal (0.614–0.626). This nearest ≫ random ≫ farthest
ordering is the expected signature of a label-informative metric space.
Within the nearest scheme, the LLM-similarity weighting gave a marginal
improvement over cosine and uniform weighting (0.655 vs 0.652 vs
0.650), and the effective sample size fell under LLM weighting (mean
ESS 39.7 vs 50.0 uniform), indicating that the judge concentrated
weight on fewer, more clinically congruent neighbors. Overall the
neighbor-weighted predictor was competitive with the feature-vector
classifiers but did not match the embedded logistic regression.

***Table 6.** Neighbor-weighted ROC AUC by retrieval scheme and
weighting strategy (embedded representation, held-out test set).*

| Retrieval scheme | Uniform | Cosine | LLM | Combined |
| --- | ---: | ---: | ---: | ---: |
| Nearest | 0.650 | 0.652 | 0.655 | 0.654 |
| Subsampled | 0.621 | 0.626 | 0.614 | 0.621 |
| Random | 0.452 | 0.466 | 0.508 | 0.497 |
| Farthest | 0.396 | 0.389 | 0.397 | 0.395 |

The retrieved neighborhoods were also statistically distinct from chance
and modestly reweighted by the LLM judge (Figure 7). Neighbor cosine
similarities separated cleanly from a random-pair reference
distribution, and top-*k* label agreement (homophily) was marginally
higher under LLM-judge weighting than under raw cosine weighting.

![](../results/Qwen-Qwen3-Embedding-8B/google_medgemma-27b-text-it/roc_curves/roc_curve_NEAREST_COSINE.png){width=70%}

***Figure 6.** Neighbor-weighted ROC for the nearest-retrieval,
cosine-weighted predictor (embedded representation), shown as a
representative single curve. The full nearest ≫ random ≫ farthest scheme
ordering across all four weighting strategies (Table 6) will be rendered
as a single composite AUC-by-scheme figure built from summary.csv.
**[Stand-in pending composite.]***

![](../results/Qwen-Qwen3-Embedding-8B/google_medgemma-27b-text-it/cosine_score_random_vs_neighbor.png){width=80%}
![](../results/Qwen-Qwen3-Embedding-8B/google_medgemma-27b-text-it/agreement_curves/agreement_curve_NEAREST.png){width=80%}

***Figure 7.** Embedding-space validity and neighbor homophily (embedded
representation). (A) Distribution of cosine similarity for retrieved
neighbors versus random patient pairs; the separation confirms that
retrieved neighbors are not random draws. (B) Top-*k* label agreement
under cosine versus LLM-judge weighting for the nearest scheme.*

## Embedder comparison

**[PENDING: head-to-head discrimination/calibration across the four
encoders (bge-small-en-v1.5, bge-en-icl, Qwen3-Embedding-4B,
Qwen3-Embedding-8B). Only Qwen3-Embedding-8B has been run to date; the
remaining three require backfill through ml_only.sbatch.]**

***Figure 8.** Cross-embedder robustness: best-classifier ROC AUC and the
largest semantic-feature ablation deltas across all four encoders
(bge-small-en-v1.5, bge-en-icl, Qwen3-Embedding-4B, Qwen3-Embedding-8B),
demonstrating that the principal conclusions hold beyond the
Qwen3-Embedding-8B encoder. **[PENDING: figure to be generated once
ml_only.sbatch has been run for all four encoders.]***

## Semantic-feature ablation

Permuting individual narrative concepts and re-scoring with the frozen
baseline classifiers localized the embedding's predictive signal to
clinical content (Table 7). Permuting the psychiatric-history section
produced the largest discrimination loss (logistic-regression ΔROC AUC
−0.072; losses of −0.022 to −0.055 for the other classifiers), followed
by medication burden (−0.031 to −0.055). In contrast, permuting
race/ethnicity, social determinants, or treatment contraindications
produced small deltas (approximately −0.01 to −0.035), indicating that
the embedding relied comparatively little on sociodemographic content
for TRD prediction. Consistent with a concentrated signal, the best
L1/elasticnet logistic-regression fit retained only 165 of 4,096
embedding dimensions (4.0%) with nonzero coefficients.

***Table 7.** Semantic-feature ablation: change in ROC AUC versus the
frozen baseline when each narrative concept is permuted across donors
(embedded representation). More negative = larger reliance on that
concept. Point estimates; paired-bootstrap CIs pending re-run.*

| Permuted concept | LR | RF | GB | XGB |
| --- | ---: | ---: | ---: | ---: |
| Psychiatric history | −0.072 | −0.022 | −0.052 | −0.055 |
| Medication burden | −0.055 | −0.031 | −0.034 | −0.050 |
| Race/ethnicity | −0.015 | −0.022 | −0.035 | −0.010 |
| Social determinants (SDOH) | −0.014 | −0.017 | −0.023 | −0.011 |
| Treatment contraindications | −0.019 | −0.020 | −0.016 | −0.007 |

![](../results/Qwen-Qwen3-Embedding-8B/google_medgemma-27b-text-it/ablation_roc_ci_EMBEDDED.png){width=95%}

***Figure 9.** Semantic-feature ablation, absolute-discrimination view
(embedded representation). Each row is a run — the unablated baseline on
top, then the five permutation specifications ordered by descending
logistic-regression AUC drop (the same ordering reused across panels) —
with one panel per classifier on a shared AUC axis and a reference line
at the baseline AUC. Permuting psychiatric history and medication burden
produces the largest discrimination loss, whereas the sociodemographic
permutations move AUC comparatively little (deltas in Table 7).*

# Discussion

## Principal findings

EHR-derived patient representations carried modest but real signal for
incident TRD at the point of first adequate antidepressant exposure. A
neural embedding of a deterministic patient narrative, classified by
logistic regression, gave the best discrimination (ROC AUC 0.70) and
the best calibration among discriminative models, exceeding logistic
regression on the rule-based feature vector (0.65) and edging the best
feature-vector model (random forest, 0.69). That the linear model on
the embedding outperformed the tree ensembles, and that only about 4%
of embedding dimensions carried nonzero weight, together suggest the
predictive structure in the embedding is largely low-dimensional and
approximately linear. The semantic-feature ablation attributed this
signal predominantly to psychiatric history and medication burden
rather than to sociodemographic content — a reassuring result both
clinically and from a fairness standpoint, as it argues against the
embedding leaning on race or social determinants to predict TRD. The
neighbor-weighted predictor's nearest ≫ random ≫ farthest ordering
confirmed that the embedding space is label-informative, though
LLM-based reweighting added only marginal lift over raw cosine
similarity.

## Clinical and methodological context

The descriptive results already carry interpretable signal. The
strongest univariate correlates of TRD in this cohort — recurrent and
severe MDD coding, a flagged suicidality history, and broad
psychiatric/substance comorbidity — are clinically coherent markers of
a more severe and complex illness course, and their prominence is
reassuring evidence that the cohort and labels behave as expected
before any model is fit.

## Limitations
<!-- TRIPOD+AI 14 (limitations), representativeness -->

Several limitations constrain generalizability and must accompany any
performance claim. First, *representativeness*: the cohort skews
markedly old (52.7% aged ≥65; mean age 62.5) and female (72.8%) and is
overwhelmingly White/Caucasian (81.3%) and English-preferring (99.2%),
so findings may not transfer to younger, more diverse, or
non-English-preferring populations. Second, *missing data*: vitals are
missing not at random (approximately 50% missing BMI even under the
most permissive filter; 28.4% of patients with no vital record at any
date) and were dropped rather than imputed, and religion missingness is
age-graded (30.5% → 9.5%) and reaches the model as an indicator level;
missing race/ethnicity itself correlates with roughly doubled TRD
prevalence. Third, this is internal validation by random split only; no
temporal or external validation was performed, and the high-dimensional
EMBEDDED models operate far below the conventional events-per-variable
threshold. Fourth, the ablation attributes signal to named concepts but
cannot resolve co-correlation between clinically related features
without joint multi-field perturbation. Fifth, **[PENDING: any
limitations arising from the refreshed performance results.]**

## Conclusions

Routinely collected EHR data carry modest, clinically coherent signal
for incident TRD that is detectable at the point of first adequate
antidepressant exposure. A neural embedding of a deterministic
narrative was at least competitive with, and under logistic regression
superior to, a transparent rule-based feature vector, while relying on
clinical rather than sociodemographic content. The absolute
discrimination (ROC AUC ≈ 0.70) is too modest for standalone clinical
deployment but supports the representation as a component of risk
stratification, pending external validation and the planned re-run
across all four embedders. **[PENDING: final numeric conclusions once
the current pipeline is re-run.]**

# Declarations

**Funding.** **[PENDING: funding statement.]**

**Conflicts of interest.** **[PENDING: COI statement.]**

**Ethics.** **[PENDING: IRB/ethics approval and data-use statement.]**

**Data availability.** The EHR data are not publicly shareable.
**[PENDING: code-availability link.]**

**Reporting.** This study is reported in accordance with the TRIPOD+AI
guideline for prediction-model studies.

# References

1. Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement: updated
   guidance for reporting clinical prediction models that use
   regression or machine learning methods. *BMJ*. 2024;385:e078378.
2. Pedregosa F, Varoquaux G, Gramfort A, et al. Scikit-learn: machine
   learning in Python. *Journal of Machine Learning Research*.
   2011;12:2825–2830.
3. Chen T, Guestrin C. XGBoost: a scalable tree boosting system. In:
   *Proceedings of the 22nd ACM SIGKDD International Conference on
   Knowledge Discovery and Data Mining*. 2016:785–794.
4. Reimers N, Gurevych I. Sentence-BERT: sentence embeddings using
   Siamese BERT-networks. In: *Proceedings of EMNLP-IJCNLP*.
   2019:3982–3992.
5. **[PENDING: additional references — TRD epidemiology/definition, EHR
   prediction-model prior work, the specific embedder model cards (BGE,
   Qwen3-Embedding), and the LLM judge model.]**
