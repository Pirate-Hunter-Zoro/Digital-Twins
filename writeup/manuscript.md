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
notebooks/figures/). Classical-ML, cross-embedder, and ablation
performance numbers are from the current pipeline run (9,724-patient
cohort, Qwen3-Embedding-8B). The neighbor-weighted results (Table 6,
Figures 4-5) are pending a full-grid re-run; see the provenance note
under Results.
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
construction (827 events against 4,096 embedding dimensions, ≈ 0.2),
and the EMBEDDED models are accordingly interpreted as
high-dimensional learners rather than as low-dimensional regression
models. On the FEATURE side the dtype-routed column transformer
expands to 91 encoded features, giving an EPV of 827/91 ≈ 9.1, just
below the conventional ≥10 threshold.

## Train/test split
<!-- TRIPOD+AI 5b / analysis -->

A single stratified 80/20 train/test split was created once and shared
across every evaluation arm to ensure fair comparison. The split
preserved the natural class imbalance: 7,779 training patients (827
TRD-positive, 10.6%) and 1,945 test patients (207 TRD-positive, 10.6%).
In the
neighbor-weighted pipeline the searchable neighbor pool was restricted
to training-set patients: every test patient served as a query anchor,
but all test identifiers were removed from the retrieval index, so each
anchor's neighbors — and therefore the labels driving its prediction —
were drawn exclusively from the training set. This prevents any test
label from informing another test patient's prediction and prevents an
anchor from retrieving itself. Train-versus-test comparability
was confirmed by per-predictor standardized mean differences (SMD),
all small in magnitude (maximum absolute SMD ≈ 0.07; Table 3).

## Model development
<!-- TRIPOD+AI 10 (model development), 11 (analysis methods) -->

**Classical machine learning.** Four classifiers — logistic regression,
random forest, gradient boosting, and XGBoost — were trained on each
representation. Each
classifier was wrapped in a pipeline with a dtype-routed column
transformer: the numeric block was standardized; the categorical block
was one-hot encoded (binary categories collapsed, unknown categories
ignored at inference); the boolean block was cast to integer and passed
through. The embedded representation is a single all-numeric block and
flows through the numeric branch only. Every classifier was tuned by
five-fold cross-validated grid search optimizing ROC AUC; the refit
best estimator supplied predicted probabilities. Because the column
transformer was bundled with the estimator in a single pipeline, all
data-dependent preprocessing — standardization parameters and one-hot
category levels — was fit only on each fold's training partition and
applied to the held-out fold, so the cross-validated estimates carry no
preprocessing leakage; with no imputation anywhere in the pipeline, the
scaler and encoder are the only fitted preprocessing steps.

**Neighbor-weighted retrieval prediction.** On the EMBEDDED
representation only, each anchor patient's predicted TRD probability was
computed from its $K = 50$ retrieved neighbors as a weighted average of
the neighbors' binary TRD labels,

$$\hat{P}(\mathrm{TRD}) = \frac{\sum_{i=1}^{K} w_i\, y_i}{\sum_{i=1}^{K} w_i},$$

where $y_i \in \{0,1\}$ is the TRD label of neighbor $i$ and $w_i$ its
weight. Four schemes defined the per-neighbor weight: uniform (every
$w_i = 1$, so the prediction is the unweighted TRD fraction among
neighbors), cosine (the anchor–neighbor cosine similarity), LLM (a
clinical-similarity score assigned to each anchor–neighbor pair by a
large language model judge and rescaled to the unit interval), and
combined (the harmonic mean of the cosine and LLM weights). As a
neighborhood-confidence diagnostic we also report the effective sample
size,

$$\mathrm{ESS} = \frac{\left(\sum_{i=1}^{K} w_i\right)^2}{\sum_{i=1}^{K} w_i^2},$$

which equals $K$ under uniform weighting and falls as weight
concentrates on fewer neighbors. The LLM judge (`MedGemma-27B`) scored
each pair by comparing the two deterministic narratives against a fixed
six-dimension clinical-similarity rubric and returning structured JSON;
only the overall 0–100 similarity, rescaled to the unit interval,
entered the weighting. The verbatim prompts, scoring rubric, and worked
high- and low-similarity examples are provided in Supplement S1. Each
weighting was evaluated under four retrieval schemes
(nearest, farthest, random, and a two-stage subsampled mode) to isolate
predictive lift against negative and random baselines. This pipeline is
embedding-only because cosine similarity is not well defined over mixed
categorical/numeric features.

**Semantic-feature ablation.** The deterministic narrative is built
from labeled sections and fields — a psychiatric-history section, a
medication-burden section, a race/ethnicity field, and so on — which
lets us probe how much the embedding's predictive signal depends on
each clinical concept individually. For one target concept at a time we
overwrote that concept's content in every patient's narrative with
content taken from another patient, re-embedded the resulting perturbed
narratives, applied the already-trained baseline classifiers without
retraining them, and measured the resulting drop in discrimination
(ΔROC AUC) against the unperturbed baseline. A large drop indicates the
embedder's representation of that concept was carrying real predictive
weight. Holding the classifiers *frozen* is deliberate: it answers the
input-perturbation question "does the embedder's encoding of concept
*X* drive predictions?" rather than "could a retrained model find a
substitute signal and route around the ablation?" — a retrained
baseline would mask the very dependence we set out to measure.

The overwrite used cohort-wide donor permutation: for a given concept,
each patient's value was replaced by that of one other patient under a
random one-to-one shuffle across the cohort. This preserves the
cohort-wide distribution of the concept (the same set of values is
present, merely reassigned to different patients) while severing the
link between each patient's concept value and their own outcome, so any
discrimination lost is attributable to that broken link rather than to
injecting artificial or out-of-distribution text. Section-level
specifications swap an entire narrative section between donor and
recipient; field-level specifications swap a single field. The
pre-specified slate comprised five specifications: race/ethnicity
(field), psychiatric history (section), medication burden (section),
treatment contraindications (section), and social determinants of
health (field). The treatment-exposure section was deliberately
excluded because prior adequate antidepressant trials are constitutive
of the TRD definition, so its ablation delta would be large,
predictable, and uninformative.

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
encoders. **[PENDING: code-availability
statement and repository/DOI link for submission.]**

# Results

## Participant flow
<!-- TRIPOD+AI 13a (participant flow), 13b (characteristics) -->

The eligibility cascade reduced a source population of 502,118 patients
to a final analysis cohort of 9,724 (Table 1). The largest single
reduction was the MDD diagnosis requirement (502,118 → 73,942),
followed by the pre-anchor history requirement (43,141 → 16,798) and
the post-anchor follow-up requirement (16,798 → 9,724). Of the final
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
| ≥2 years pre-anchor history | 16,798 | 26,343 |
| ≥1 year post-anchor follow-up (final cohort) | 9,724 | 7,074 |

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
| Train | 7,779 | 827 | 10.6% |
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
rather than a richness artifact. The TRD-stratified distribution figures
are provided in the supplement.

## Provenance of performance results

The classical-ML discrimination and calibration results, the
cross-embedder comparison, and the semantic-feature ablation below are
from the current pipeline run on the 9,724-patient cohort, sharing the
1,945-patient held-out test split used for the cohort characterization
above, with the `Qwen3-Embedding-8B` encoder and a `MedGemma-27B`
similarity judge. The FEATURE-side classical-ML numbers were computed
on the production 91-column matrix (the three cohort-averaged vital
columns and a legacy psychotherapy-count column have been removed; no
NaN cells reach any transformer), and the feature vector is
embedder-independent, so the FEATURE numbers are identical across
encoders. Ablation deltas carry paired-bootstrap delta-AUC confidence
intervals. One section is not yet refreshed: the neighbor-weighted
results (Table 6, Figures 4-5) are reported from the full-grid
retrieval run (nearest / farthest / random / subsampled × four
weighting strategies, with the LLM judge active); a later trimmed
re-run overwrote the machine-readable neighbor outputs, so those
values are being regenerated and will be updated on completion.
**[PENDING: refresh Table 6 / Figures 4-5 from the full-grid 8B KNN
re-run.]**

## Model discrimination
<!-- TRIPOD+AI 13b -->

On the embedded representation, logistic regression achieved the
highest discrimination (ROC AUC 0.688, 95% CI 0.647–0.724; AUPRC
0.248), followed by XGBoost (0.680), random forest (0.667), and
gradient boosting (0.660). On the rule-based feature vector, random
forest led (0.691, 95% CI 0.649–0.730), with XGBoost close behind
(0.690) and logistic regression and gradient boosting at 0.688 and
0.676. The two representations were thus essentially equivalent in
discrimination on this encoder: embedded logistic regression (0.688)
matched its feature-vector counterpart exactly (0.688) and fell
marginally short of the best feature-vector model (random forest,
0.691), with all confidence intervals broadly overlapping. Every
model's discrimination remained modest — consistent with a difficult
target and a 10.6% positive rate (Table 4). Embedded logistic
regression was the strongest classifier on the embedded representation
for every encoder evaluated, and on the Qwen3-Embedding-4B encoder it
exceeded the best feature-vector model (0.703 vs 0.691; Figure 6).

***Table 4.** Discrimination of the four classifiers on each
representation (held-out test set). 95% CIs are bootstrap percentile
intervals. FEATURE-side values are from the production 91-column matrix
and are embedder-independent (see provenance note).*

| Representation | Classifier | ROC AUC (95% CI) | AUPRC |
| --- | --- | :---: | ---: |
| EMBEDDED | Logistic regression | 0.688 (0.647–0.724) | 0.248 |
| EMBEDDED | Random forest | 0.667 (0.629–0.707) | 0.219 |
| EMBEDDED | Gradient boosting | 0.660 (0.620–0.700) | 0.230 |
| EMBEDDED | XGBoost | 0.680 (0.642–0.717) | 0.232 |
| FEATURE | Logistic regression | 0.688 (0.649–0.727) | 0.235 |
| FEATURE | Random forest | 0.691 (0.649–0.730) | 0.248 |
| FEATURE | Gradient boosting | 0.676 (0.634–0.716) | 0.228 |
| FEATURE | XGBoost | 0.690 (0.650–0.731) | 0.242 |

![](../results/Qwen-Qwen3-Embedding-8B/google_medgemma-27b-text-it/roc_curves/roc_curve_logistic_regression_EMBEDDED.png){width=80%}
![](../results/Qwen-Qwen3-Embedding-8B/google_medgemma-27b-text-it/roc_curves/roc_curve_random_forest_FEATURE.png){width=80%}

***Figure 1.** Receiver-operating-characteristic curves for the best
classifier on each representation (held-out test set). (A) Embedded
logistic regression; (B) rule-based random forest. Shaded bands are
bootstrap 95% confidence intervals; point AUCs are given in Table 4.
The cross-embedder discrimination comparison is shown in Figure 6.*

![](../results/Qwen-Qwen3-Embedding-8B/google_medgemma-27b-text-it/confusion_matrices/confusion_matrix_logistic_regression_EMBEDDED.png){width=80%}
![](../results/Qwen-Qwen3-Embedding-8B/google_medgemma-27b-text-it/confusion_matrices/confusion_matrix_random_forest_FEATURE.png){width=80%}

***Figure 2.** Confusion matrices at the Youden-J optimal operating point
for the best classifier on each representation (held-out test set).
(A) Embedded logistic regression; (B) rule-based random forest.*

## Model calibration

Calibration varied by classifier and representation (Table 5). On the
embedded representation, random forest was the best-calibrated model
(slope 0.95, weighted calibration error 0.006); logistic regression was
mildly under-confident (slope 1.17, its slope above 1 indicating
probabilities pulled toward the base rate) but carried low absolute
error (Brier 0.090), while gradient boosting and XGBoost showed steeper
slopes (1.86 and 1.58). On the feature vector, logistic regression was
well-behaved (slope 0.80, weighted calibration error 0.006). Brier
scores were uniformly near 0.089–0.091 across all eight models,
reflecting the low base rate. No model placed any test prediction above
0.9, and a large fraction fell below 0.1, as expected.

***Table 5.** Calibration of the four classifiers on each
representation (held-out test set). Brier and weighted calibration
error (WCE): lower is better; calibration slope: ideal is 1; intercept:
ideal is 0.*

| Representation | Classifier | Brier | WCE | Slope | Intercept |
| --- | --- | ---: | ---: | ---: | ---: |
| EMBEDDED | Logistic regression | 0.090 | 0.020 | 1.17 | −0.07 |
| EMBEDDED | Random forest | 0.091 | 0.006 | 0.95 | 0.02 |
| EMBEDDED | Gradient boosting | 0.091 | 0.007 | 1.86 | −0.14 |
| EMBEDDED | XGBoost | 0.090 | 0.015 | 1.58 | −0.09 |
| FEATURE | Logistic regression | 0.090 | 0.006 | 0.80 | 0.04 |
| FEATURE | Random forest | 0.089 | 0.014 | 1.39 | −0.05 |
| FEATURE | Gradient boosting | 0.090 | 0.019 | 1.16 | −0.08 |
| FEATURE | XGBoost | 0.090 | 0.020 | 1.15 | −0.01 |

Calibration curves for all classifiers are provided in the supplement.

## Feature importance

Feature-level interpretability is available only for the rule-based
representation: the embedded representation's 4,096 latent dimensions
carry no individual clinical meaning, so per-concept attribution on the
embedded side is addressed by the semantic-feature ablation below rather
than by a feature-importance ranking. On the rule-based representation,
the highest-weighted predictors were clinically coherent and
recapitulated the strongest univariate correlates of TRD (Table 2):
obsessive–compulsive disorder, a flagged suicidality history, any
substance-use disorder, and severe MDD coding carried the largest
positive logistic-regression weights, while a smaller set of indicators
(e.g., nicotine use disorder, an employment-related social-determinant
flag) carried negative weights (Figure 3).

![](../results/Qwen-Qwen3-Embedding-8B/google_medgemma-27b-text-it/feature_importance/feature_importance_logistic_regression.png){width=80%}
![](../results/Qwen-Qwen3-Embedding-8B/google_medgemma-27b-text-it/feature_importance/feature_importance_random_forest.png){width=80%}

***Figure 3.** Feature importance on the rule-based representation.
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
weighting strategy (embedded representation, held-out test set). Values
are from the full-grid retrieval run and are being regenerated; see the
provenance note. **[PENDING: refresh from the full-grid 8B KNN re-run.]***

| Retrieval scheme | Uniform | Cosine | LLM | Combined |
| --- | ---: | ---: | ---: | ---: |
| Nearest | 0.650 | 0.652 | 0.655 | 0.654 |
| Subsampled | 0.621 | 0.626 | 0.614 | 0.621 |
| Random | 0.452 | 0.466 | 0.508 | 0.497 |
| Farthest | 0.396 | 0.389 | 0.397 | 0.395 |

![](../results/Qwen-Qwen3-Embedding-8B/google_medgemma-27b-text-it/roc_curves/roc_curve_NEAREST_LLM.png){width=80%}
![](../results/Qwen-Qwen3-Embedding-8B/google_medgemma-27b-text-it/roc_curves/roc_curve_RANDOM_LLM.png){width=80%}

***Figure 4.** Neighbor-weighted ROC for the LLM-weighted predictor on
the embedded representation (held-out test set). (A) Nearest retrieval;
(B) random retrieval. The separation between the nearest and random
schemes is the discriminative signal; weighting-strategy and the
farthest/subsampled schemes are reported in Table 6 and the supplement.*

![](../results/Qwen-Qwen3-Embedding-8B/google_medgemma-27b-text-it/confusion_matrices/confusion_matrix_NEAREST_LLM.png){width=80%}
![](../results/Qwen-Qwen3-Embedding-8B/google_medgemma-27b-text-it/confusion_matrices/confusion_matrix_RANDOM_LLM.png){width=80%}

***Figure 5.** Confusion matrices at the Youden-J optimal operating point
for the LLM-weighted neighbor predictor (embedded representation).
(A) Nearest retrieval; (B) random retrieval.*

## Embedder comparison

Across the four encoders the embedded logistic-regression model
occupied a narrow discrimination band (ROC AUC 0.684–0.703; Table 8,
Figure 6A). Qwen3-Embedding-4B was highest (0.703, 95% CI 0.663–0.740)
and bge-small-en-v1.5 lowest (0.684); the larger Qwen3-Embedding-8B
(0.688) did not improve on the 4B variant. Logistic regression was the
best-discriminating classifier on the embedded representation for every
encoder. The semantic-feature ablation reproduced across encoders
(Figure 6B): permuting psychiatric history and medication burden
produced the largest discrimination losses for all four (psychiatric
history ΔROC AUC −0.033 to −0.050; medication burden −0.037 to −0.048),
each with paired-bootstrap intervals excluding zero, whereas the
sociodemographic permutations moved AUC little. The principal
conclusions — modest discrimination led by the linear model, and
reliance on clinical rather than sociodemographic content — therefore
hold beyond the Qwen3-Embedding-8B encoder.

***Table 8.** Embedded logistic-regression discrimination by encoder
(held-out test set). 95% CIs are bootstrap percentile intervals.*

| Encoder | ROC AUC (95% CI) | AUPRC |
| --- | :---: | ---: |
| bge-small-en-v1.5 | 0.684 (0.644–0.723) | 0.204 |
| bge-en-icl | 0.690 (0.649–0.728) | 0.232 |
| Qwen3-Embedding-4B | 0.703 (0.663–0.740) | 0.236 |
| Qwen3-Embedding-8B | 0.688 (0.647–0.724) | 0.248 |

![](../results/cross_embedder_robustness_EMBEDDED.png){width=95%}

***Figure 6.** Cross-embedder robustness: embedded logistic-regression
ROC AUC (A) and the two largest semantic-feature ablation deltas
(psychiatric history, medication burden; B) across all four encoders
(bge-small-en-v1.5, bge-en-icl, Qwen3-Embedding-4B, Qwen3-Embedding-8B),
demonstrating that the principal conclusions hold beyond the
Qwen3-Embedding-8B encoder. Error bars are bootstrap (A) and
paired-bootstrap (B) 95% confidence intervals.*

## Semantic-feature ablation

Permuting individual narrative concepts and re-scoring with the frozen
baseline classifiers localized the embedding's predictive signal to
clinical content (Table 7). Permuting the psychiatric-history section
produced the largest discrimination loss (logistic-regression ΔROC AUC
−0.042; losses of −0.036 to −0.053 for the other classifiers), followed
by medication burden (−0.010 to −0.044). In contrast, permuting
race/ethnicity, social determinants, or treatment contraindications
produced negligible deltas (within ±0.008 for every classifier),
indicating that the embedding relied comparatively little on
sociodemographic content for TRD prediction. Paired-bootstrap
delta-AUC confidence intervals excluded zero for psychiatric history
across all four classifiers and for medication burden in three of four
(gradient boosting's interval included zero); none of the
sociodemographic permutations produced an interval excluding zero. The
embedded signal was not sparse: the best logistic-regression fit
retained all 4,096 embedding dimensions with nonzero coefficients, and
its cumulative-importance curve was diffuse (80% of the coefficient
magnitude was spread across roughly 2,067 of 4,096 dimensions, and 90%
across 2,659), indicating that TRD-relevant information is distributed
broadly across the embedding rather than concentrated in a few
dimensions.

***Table 7.** Semantic-feature ablation: change in ROC AUC versus the
frozen baseline when each narrative concept is permuted across donors
(embedded representation). More negative = larger reliance on that
concept. Point estimates; paired-bootstrap delta-AUC CIs are shown in
Figures 6B and 7.*

| Permuted concept | LR | RF | GB | XGB |
| --- | ---: | ---: | ---: | ---: |
| Psychiatric history | −0.042 | −0.036 | −0.047 | −0.053 |
| Medication burden | −0.037 | −0.022 | −0.010 | −0.044 |
| Race/ethnicity | −0.002 | −0.003 | 0.000 | −0.003 |
| Social determinants (SDOH) | −0.003 | +0.002 | −0.002 | −0.002 |
| Treatment contraindications | 0.000 | −0.004 | +0.005 | −0.008 |

![](../results/Qwen-Qwen3-Embedding-8B/google_medgemma-27b-text-it/ablation_roc_ci_EMBEDDED.png){width=95%}

***Figure 7.** Semantic-feature ablation, absolute-discrimination view
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
logistic regression, achieved discrimination essentially equivalent to
the best rule-based model on the primary encoder (embedded logistic
regression ROC AUC 0.688 versus rule-based random forest 0.691;
embedded and rule-based logistic regression were identical at 0.688).
Logistic regression was the strongest embedded classifier on all four
encoders evaluated, and on the Qwen3-Embedding-4B encoder embedded
logistic regression (0.703) exceeded the best rule-based model; that a
linear model on the embedding matched or beat the tree ensembles
suggests the predictive structure is approximately linear. The signal
was not, however, low-dimensional — the logistic-regression fit used
all 4,096 embedding dimensions, with 80% of the coefficient magnitude
spread across roughly half of them, so the embedding distributes
TRD-relevant information broadly rather than concentrating it in a few
latent directions. The
semantic-feature ablation attributed this
signal predominantly to psychiatric history and medication burden
rather than to sociodemographic content — a reassuring result both
clinically and from a fairness standpoint, as it argues against the
embedding leaning on race or social determinants to predict TRD. The
neighbor-weighted predictor's nearest ≫ random ≫ farthest ordering
confirmed that the embedding space is label-informative, though
LLM-based reweighting added only marginal lift over raw cosine
similarity — consistent with the weighting strategy being second-order
to the retrieval scheme, since discrimination tracked which neighbors
were retrieved rather than how they were weighted. The judge's overall
similarity scores were well-behaved and were the only quantity entering
the weighting; its per-dimension sub-scores and free-text rationales, by
contrast, were unreliable (they referenced symptom data absent from the
narrative and occasionally flagged identical fields as mismatches;
Supplement S1), so we treat the sub-scores as diagnostic only.

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
without joint multi-field perturbation. Fifth, the embedded
representation did not deliver a discrimination advantage over the
transparent rule-based feature vector on the primary encoder (the two
were within 0.003 ROC AUC and their logistic-regression models were
identical); the embedding's appeal therefore rests on achieving parity
without hand-engineered features and on consistency across encoders,
not on an accuracy gain.

## Conclusions

Routinely collected EHR data carry modest, clinically coherent signal
for incident TRD that is detectable at the point of first adequate
antidepressant exposure. A neural embedding of a deterministic
narrative achieved discrimination essentially equivalent to a
transparent rule-based feature vector (both ROC AUC ≈ 0.69 on the
primary encoder), with logistic regression the strongest embedded model
across all four encoders evaluated and relying on clinical rather than
sociodemographic content. The absolute discrimination (ROC AUC ≈ 0.69)
is too modest for standalone clinical deployment but supports the
representation as a component of risk stratification, pending external
validation.

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
