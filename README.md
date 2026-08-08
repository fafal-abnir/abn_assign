# Fraud Detection with Ensemble and Deep Tabular Models

The project compares tree ensembles, imbalance-aware boosting methods, FT-Transformer, and TabM under a common temporal evaluation protocol.

The implementation supports:

- time-based validation and untouched final-test evaluation
- leakage-aware historical transaction features
- multiple feature-engineering strategies
- random under-sampling, random over-sampling, and SMOTE
- EasyEnsemble and RUSBoost with internal under-sampling
- randomized hyperparameter search for ensemble methods
- weighted BCE, focal loss, and deviation-regularized BCE for deep models
- optional SHAP and native feature-importance reports
- experiment tracking through timestamped `metrics.json` files
- notebook-based comparison of ensemble and deep-learning experiments.

---

## Overview

Fraud detection is a highly imbalanced binary-classification problem. Accuracy alone is therefore not informative: a model can classify almost every transaction as legitimate and still achieve high accuracy.

This project uses **average precision** as the validation-selection metric. ROC-AUC and average precision are reported on the final test set, but test performance is not used to choose the model, feature strategy, sampling strategy, loss, or number of epochs.

The workflow is:

1. Sort `fraudTrain.csv` chronologically.
2. Use the latest fraction of the training file as temporal validation data.
3. Fit preprocessing, resampling, and model parameters on the earlier period.
4. Select the configuration using validation average precision.
5. Refit the selected configuration on all rows from `fraudTrain.csv`.
6. Evaluate once on the later `fraudTest.csv` period.

---

## Supported Models

### Ensemble methods

- **XGBoost**
- **Random Forest**
- **CatBoost**
- **LightGBM**
- **EasyEnsemble**
- **RUSBoost**

XGBoost, Random Forest, CatBoost, and LightGBM can be combined with the external sampling strategies described below. EasyEnsemble and RUSBoost already perform under-sampling internally and must be run with `--sampling none`.

### Deep-learning methods

- **FT-Transformer**
- **TabM**

Both models are implemented with PyTorch and Lightning. Categorical variables use learned embeddings, while numerical features are standardized using statistics fitted only on the applicable training period.

---

## Feature Strategies

Every strategy includes the base categorical and numerical features.

### Base

Categorical features:

- merchant
- transaction category
- gender
- state
- job.

Numerical features:

- transaction amount
- city population
- age at transaction time
- log-transformed amount
- round-amount indicator.

### Time

Adds:

- hour
- day of week
- month
- weekend indicator.

### Geographic

Adds:

- customer and merchant coordinates
- customer-to-merchant distance
- likely-different-state indicator
- distance from the previous merchant location.

### Behavioral

Adds causal card-history features:

- prior transaction count
- observed card age
- hours since the previous transaction
- amount relative to the previous amount
- amount relative to the historical mean

### All

Combines base, time, geographic, and behavioral features.

Feature-strategy identifiers accepted by the command-line interfaces are:

```text
base
time
geo
behavioral
all
```

---

## Imbalance Strategies

The ensemble workflow supports:

```text
none
random_under
random_over
smote
```

Sampling is part of the imbalanced-learn pipeline and is applied only to training folds. Validation and test observations are never resampled.

When sampling is `none`, supported tree models use class weighting. EasyEnsemble and RUSBoost manage imbalance internally, so external sampling is deliberately disabled for them.

Deep models support three loss functions:

- `weighted_bce`: binary cross-entropy weighted by the training-period class ratio;
- `focal`: focal loss for emphasizing difficult examples
- `weighted_bce+deviation`: weighted BCE plus the deviation loss described in [DevNet](https://arxiv.org/abs/1911.08623).

The deviation component standardizes anomaly scores against normally distributed reference samples. Its weight, margin, and reference-sample count are configurable.

---

## Repository Structure

```text
ABNAmro_assignment/
├── fraud_detection/
│   ├── config.py             # Models, feature groups, sampling, and run configuration
│   ├── data.py               # Loading, causal feature engineering, and preprocessing
│   ├── models.py             # Ensemble estimators and hyperparameter spaces
│   ├── training.py           # Single ensemble experiment
│   ├── search.py             # Validation search across complete configurations
│   └── explainability.py     # SHAP and native feature-importance outputs
├── deep_learning/
│   ├── config.py             # Deep-model and loss configuration
│   ├── data.py               # Train-fitted encoding and Lightning data loaders
│   ├── models.py             # FT-Transformer and TabM
│   ├── module.py             # Losses, optimization, and metrics
│   └── training.py           # Epoch selection, full refit, and test evaluation
├── notebook/
│   ├── data_exploration.ipynb
│   └── experiment_analysis.ipynb
├── train_ensemble.py         # Single ensemble CLI
├── search_ensemble.py        # Multi-configuration ensemble search CLI
├── train_deep.py             # Deep-learning CLI
├── commands                  # Explicit experiment commands
├── pyproject.toml
└── poetry.lock
```

---

## Getting Started

### Prerequisites

- Python `>=3.12,<3.14`
- Poetry
- sufficient memory for one-hot encoded ensemble experiments
- CUDA-capable GPU recommended, but not required, for deep-learning runs.

### Installation

Clone the repository and enter it:

```bash
git clone https://github.com/fafal-abnir/abn_assign
cd ABNAmro_assignment
```

Select Python 3.12 with pyenv if necessary:

```bash
pyenv install -s 3.12.11
pyenv local 3.12.11
poetry env use "$(pyenv which python)"
```

Install dependencies. This repository is an experiment project rather than an installable Python package, so use `--no-root`:

```bash
poetry install --no-root
```

Run commands through Poetry:

```bash
poetry run python train_ensemble.py --help
poetry run python search_ensemble.py --help
poetry run python train_deep.py --help
```

---

## Data Preparation

Place the two CSV files in one directory:

```text
data/
├── fraudTrain.csv
└── fraudTest.csv
```

Both files must contain the expected transaction columns, including:

- `trans_date_trans_time`
- `is_fraud`
- `cc_num`
- `merchant`, `category`, `gender`, `state`, and `job`
- `amt`, `city_pop`, and `dob`
- `lat`, `long`, `merch_lat`, and `merch_long`.

The loader verifies that the test period starts after the training period. Each file is sorted chronologically before splitting and feature generation.

---

## Training an Ensemble Model

Train and tune one model, feature strategy, and sampling strategy:

```bash
poetry run python train_ensemble.py \
  --model lightgbm \
  --data-dir data \
  --feature-strategy all \
  --sampling none \
  --validation-fraction 0.2 \
  --n-iter 10 \
  --random-state 42;
```

Run EasyEnsemble or RUSBoost without an external sampler:

```bash
poetry run python train_ensemble.py \
  --model easyensemble \
  --data-dir data \
  --feature-strategy behavioral \
  --sampling none \
  --n-iter 10;
```

Enable explanation outputs when required:

```bash
poetry run python train_ensemble.py \
  --model xgboost \
  --data-dir data \
  --feature-strategy all \
  --sampling none \
  --n-iter 10 \
  --feature-importance \
  --shap;
```

---

## Searching Ensemble Configurations

`search_ensemble.py` compares complete model, feature, sampling, and hyperparameter configurations using only the temporal validation period. It evaluates the winning configuration on the test set.

Search all supported configurations:

```bash
poetry run python search_ensemble.py \
  --data-dir data \
  --validation-fraction 0.2 \
  --n-iter 10;
```

Search a selected subset:

```bash
poetry run python search_ensemble.py \
  --models xgboost lightgbm easyensemble rusboost \
  --feature-strategies base time behavioral all \
  --sampling-strategies none random_under random_over smote \
  --data-dir data \
  --n-iter 10;
```

For EasyEnsemble and RUSBoost, the search automatically keeps only `sampling=none` candidates because sampling occurs inside those estimators.

---

## Training Deep Models

FT-Transformer example:

```bash
poetry run python train_deep.py \
  --model ft_transformer \
  --data-dir data \
  --feature-strategy all \
  --loss focal \
  --batch-size 2048 \
  --max-epochs 30 \
  --patience 5 \
  --learning-rate 0.0001;
```

TabM with weighted BCE and deviation loss:

```bash
poetry run python train_deep.py \
  --model tabm \
  --data-dir data \
  --feature-strategy all \
  --loss 'weighted_bce+deviation' \
  --batch-size 2048 \
  --max-epochs 30 \
  --patience 5 \
  --learning-rate 0.0001 \
  --deviation-weight 0.25 \
  --deviation-margin 5 \
  --deviation-reference-samples 5000;
```

During the first stage, early stopping selects the epoch count using validation average precision. A fresh model and preprocessor are then fitted on all of `fraudTrain.csv` for the selected number of epochs before final test evaluation.

---

## Main Arguments

### Ensemble training

| Group | Arguments and defaults |
|---|---|
| Model | `--model` required |
| Data | `--data-dir` required, `--output-dir experiments` |
| Features | `--feature-strategy base` |
| Imbalance | `--sampling none` |
| Validation | `--validation-fraction 0.2` |
| Search | `--n-iter 10`, `--random-state 42` |
| Explanation | `--shap`, `--feature-importance` disabled by default |

### Deep training

| Group | Arguments and defaults |
|---|---|
| Model | `--model {ft_transformer,tabm}` required |
| Features | `--feature-strategy all` |
| Loss | `--loss weighted_bce` |
| Validation | `--validation-fraction 0.2` |
| Training | `--batch-size 2048`, `--max-epochs 30`, `--patience 5` |
| Optimization | `--learning-rate 1e-3`, `--weight-decay 1e-5` |
| Architecture | `--d-token 64`, `--n-layers 3`, `--n-heads 8`, `--dropout 0.1`, `--tabm-k 16` |
| Focal loss | `--focal-alpha 0.95`, `--focal-gamma 2.0` |
| Deviation loss | `--deviation-weight 1`, `--deviation-margin 5`, `--deviation-reference-samples 5000` |
| Runtime | `--num-workers 0`, `--random-state 42`, `--output-dir experiments` |

Use each script's `--help` output as the authoritative complete interface.

---

## Experiment Outputs

Single ensemble runs are written to:

```text
experiments/<model>/<timestamp>/metrics.json
```

Automatic ensemble searches are written below:

```text
experiments/search/<timestamp>/
```

Deep-learning runs are written to:

```text
experiments/deep/<model>/<timestamp>/metrics.json
```

The timestamp includes the local UTC offset. Metrics files contain the selected configuration, validation average precision, test ROC-AUC, test average precision, and relevant split timestamps. Deep metrics also contain the selected epoch count and full training configuration.

The training code intentionally does not save fitted pipelines, model checkpoints, or prediction CSV files.

---

## Experiment Analysis

Open `notebook/experiment_analysis.ipynb` to:

- parse all timestamped metrics files recursively
- rank ensemble experiments by validation average precision
- rank deep-learning experiments separately
- inspect the best configuration overall and per model
- compare the influence of ensemble feature and sampling strategies
- compare the influence of deep feature strategies and loss functions
- visualize individual runs and aggregate distributions.

The plots use validation average precision. Test results are displayed for final reporting and should not be used to choose among configurations.

`notebook/data_exploration.ipynb` contains exploratory analysis of the transaction data and train/test identity overlap.

---

## Temporal Leakage and Reproducibility

For a transaction at time $t_i$:

- historical card features use shifted or cumulative statistics from earlier transactions
- the current transaction is excluded from its own historical mean
- training and test files must occupy non-overlapping chronological periods
- the validation set is the latest portion of `fraudTrain.csv`
- external resampling is performed inside training folds only
- ensemble preprocessing is fitted through the training pipeline
- deep categorical vocabularies and numerical scaling statistics are fitted on the earlier training period during epoch selection
- the final test set does not determine hyperparameters, sampling, features, loss, or epoch count

The test period contains mostly customers already observed during training. This represents future-transaction prediction for known customers, not evaluation on an entirely unseen-customer population. A separate group split by `cc_num` would be required to measure generalization to new customers.

Random seeds are configurable. Lightning deterministic mode is enabled, although exact reproducibility can still depend on library versions, hardware, and device-specific operations.

---

## Metrics

The primary metric is:

- **Average precision (AP):** suitable for ranking performance under strong class imbalance.

The final report also includes:

- **ROC-AUC:** ranking performance across classification thresholds;
- **classification report:** precision, recall, and F1 score at the default probability threshold for ensemble runs.

Always compare configurations using validation AP before inspecting final-test performance.
