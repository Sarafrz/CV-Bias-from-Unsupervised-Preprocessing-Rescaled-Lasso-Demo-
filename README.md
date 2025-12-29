# Cross-Validation Bias from Unsupervised Preprocessing
## A Complete Reproducible Demo Following Moscovich & Rosset (JRSS-B)

---

## Reference Paper

This code is written to explicitly follow the methodology, assumptions, and warnings
presented in the following paper:

Amit Moscovich, Saharon Rosset (2022)  
On the cross-validation bias due to unsupervised preprocessing  
Journal of the Royal Statistical Society: Series B (JRSS-B)

---

## Motivation

A widespread belief in machine learning practice is:

- Using the target variable y before cross-validation causes data leakage  
- Using only the input features X before cross-validation is always safe  

The referenced paper rigorously proves that this belief is incorrect.

Even unsupervised preprocessing, which does not use y, can introduce
cross-validation bias if it is learned using the full dataset prior to CV.

This repository provides a minimal, transparent, and reproducible
implementation of that phenomenon.

---

## Core Statistical Principle

Cross-validation relies on the assumption that validation data must be
statistically independent of everything learned during training,
including preprocessing steps.

Therefore, any data-dependent transformation must be estimated only from
the training fold inside each CV split.

---

## The Two Pipelines Compared

This code compares two pipelines that differ in exactly one aspect:
where preprocessing is learned.

Correct pipeline (no leakage):
- preprocessing is part of a Pipeline
- the scaler is fit separately inside each CV fold
- validation data never influences preprocessing

Incorrect pipeline (information leakage):
- preprocessing is fit once on the full dataset
- cross-validation is applied afterward
- validation samples influence feature scaling

---

## High-Level Experimental Workflow

For each repetition:

1. Generate a synthetic regression dataset (X, y)
2. Generate a large independent holdout set (Xh, yh)
3. Perform 10-fold cross-validation to select the Lasso regularization parameter alpha
   - once using the correct pipeline
   - once using the incorrect pipeline
4. Evaluate both selected models on the holdout set
5. Record selected alpha and holdout mean squared error (MSE)

This process is repeated multiple times to observe systematic effects.

---

## Data Generation Details

The data-generating process follows a heavy-tailed setting:

- Feature matrix:
  X ~ Student-t(df)

- Regression coefficients:
  beta ~ Student-t(df)

- Response:
  y = X @ beta + Gaussian noise

This setup increases variability and makes the effect of preprocessing
on model selection more visible.

---

## Preprocessing Step: Feature Rescaling

Each feature is rescaled using:

sigma_j = sqrt(mean(x_j^2))

Important properties of this transformation:

- It does not use the target variable y
- It is data-dependent
- It must therefore be fit inside cross-validation to avoid leakage

A custom scikit-learn transformer is implemented to make this behavior explicit.

---

## Correct Pipeline (Recommended by the Paper)

Example structure:

correct_pipe = Pipeline([
    ("scaler", UnitVarScalerAssumingZeroMean()),
    ("model", Lasso(fit_intercept=False))
])

Grid search is then applied to this pipeline.

Why this is correct:

- the scaler is refit independently in each CV fold
- validation data never influences preprocessing
- theoretical guarantees of cross-validation are preserved

---

## Incorrect Pipeline (Demonstrating the Bias)

Example structure:

scaler = UnitVarScalerAssumingZeroMean()  
X_scaled = scaler.fit_transform(X)

Grid search is then applied to the already-scaled data.

Why this is incorrect:

- validation samples affect the estimated scaling factors
- CV independence assumptions are violated
- bias is introduced in:
  - error estimation
  - hyperparameter selection

This is exactly the pitfall analyzed in the paper.

---

## Evaluation Strategy

A large independent holdout set is used:

- never used in preprocessing
- never used in cross-validation
- provides an approximately unbiased estimate of generalization error

This allows a clean comparison between pipelines.

---

## Reported Outputs

For each experiment, the script reports:

Correct mean MSE  
Incorrect mean MSE  
Penalty = Incorrect − Correct  

It also produces two plots:

1. Histogram of selected alpha values (Correct vs Incorrect)
2. Boxplot of holdout MSE (lower is better)

---

## Installation

pip install numpy matplotlib scikit-learn

---

## How to Run

Run the script directly:

python your_script_name.py

The demo is executed at the bottom of the file:

run_demo(
    seed=7331,
    reps=25,
    N=120,
    p=300,
    df=4,
    sigma=10.0
)

---

## Interpretation and Takeaway

Key lesson:

Using only X does NOT guarantee safety.  
Any preprocessing step that learns from data must be included inside CV.

Even simple rescaling can:
- change the selected model
- alter generalization performance
- introduce subtle but real bias

---

## Intended Use

- Teaching cross-validation theory
- Classroom demonstrations
- Reproducing JRSS-B insights
- Sanity-checking machine learning pipelines

---

## Final Remark

This code is intentionally minimal and pedagogical.
Its purpose is not performance optimization, but to make
a subtle statistical issue visible, concrete, and reproducible,
exactly as emphasized by Moscovich and Rosset.
