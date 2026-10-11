# NYC Airbnb Rental Price Pipeline

An end-to-end machine learning pipeline for estimating nightly rental prices in New York City. This project was completed using the Udacity ML pipeline starter kit. MLflow runs the components, Hydra manages configuration, and Weights & Biases (W&B) tracks runs, metrics, and versioned artifacts.

## Submission links

- **GitHub repository:** https://github.com/SteffanyMcH2os/Project-Build-an-ML-Pipeline-Starter
- **W&B project:** https://wandb.ai/smcwate-western-governors-university/nyc_airbnb 
- **GitHub releases:** https://github.com/SteffanyMcH2os/Project-Build-an-ML-Pipeline-Starter/releases

The W&B project contains the datasets, training runs, model artifacts, and their lineage. Public access must be enabled before submission.

## Project structure

| File or folder | Purpose |
| --- | --- |
| `MLproject` | Defines the entry point for the overall pipeline. |
| `main.py` | Selects and launches pipeline steps. |
| `config.yaml` | Defines the sample, cleaning limits, data checks, splitting settings, and model hyperparameters. |
| `environment.yml` | Defines the development Conda environment. |
| `src/eda/eda.ipynb` | Exploratory data analysis and visualizations. |
| `src/basic_cleaning/` | Filters prices, converts review dates, and filters geographic coordinates. |
| `src/data_check/` | Tests the cleaned dataset. |
| `components/train_val_test_split/` | Creates training/validation and held-out test datasets. |
| `src/train_random_forest/` | Builds preprocessing and regression pipelines, evaluates validation performance, and exports the model. |
| `components/test_regression_model/` | Evaluates the selected production model on the held-out test set. |
| `components/get_data/` | Obtains the selected sample and uploads the raw artifact. |

Each component's `MLproject` defines its parameters and launch command; its `conda.yml` defines its execution environment.

## Setup and execution

Install Conda, clone the repository, and run these commands from the repository root—the folder containing `main.py`, `MLproject`, and `config.yaml`:

```bash
git clone https://github.com/SteffanyMcH2os/Project-Build-an-ML-Pipeline-Starter.git
cd Project-Build-an-ML-Pipeline-Starter
conda env create -f environment.yml
conda activate nyc_airbnb_dev
wandb login
```

The development environment specifies Python 3.13, MLflow 3.3.2, and W&B 0.24.0. Authenticate with your own W&B account. To log to your own project, update `main.project_name` in `config.yaml` and select your W&B entity as needed.

Run the default pipeline:

```bash
mlflow run .
```

The default steps are download, basic cleaning, data checks, data splitting, and random forest training. Test-set evaluation is selected explicitly after assigning the chosen model a `prod` alias.

Run selected steps:

```bash
mlflow run . -P steps=download,basic_cleaning
mlflow run . -P steps=data_check
mlflow run . -P steps=train_random_forest
mlflow run . -P steps=test_regression_model
```

A step run on its own requires its input artifacts to exist in W&B. Before running data checks, assign the original cleaned dataset the `reference` alias. Before test-set evaluation, assign the selected model the `prod` alias.

Run the four-combination hyperparameter search:

```bash
mlflow run . -P steps=train_random_forest -P hydra_options="modeling.random_forest.max_depth=10,50 modeling.random_forest.n_estimators=100,200 -m"
```

## Exploratory analysis

The initial sample showed:

- Seven missing listing names and eight missing host names.
- 4,123 missing values in both `last_review` and `reviews_per_month`; all corresponding records had zero reviews.
- Listing names contain descriptive text rather than just a property or owner name.
- Several numeric features have long upper tails, including price, minimum nights, review counts, and host listing counts.
- Zero availability occurs in the data, but the dataset alone does not establish why a listing has no available days.

Boxplots were used to examine distributions. Some plots hide outlier markers to make the boxes and whiskers visible; this does not remove observations from the dataset. Cleaning converts review dates to datetime and retains prices from $10 to $350, inclusive. Release 1.0.1 also retains only coordinates within longitude [-74.25, -73.50] and latitude [40.5, 41.2].

## Data checks and artifacts

The six data checks validate column names and order, borough names, geographic boundaries, borough distribution against a reference sample, row count, and price range. The row count must be strictly between 15,000 and 1,000,000. The distribution check uses a KL-divergence threshold of 0.2.

| W&B artifact | Purpose |
| --- | --- |
| `sample.csv` | Raw sample, uploaded with type `raw_data`. |
| `clean_sample.csv` | Cleaned sample, uploaded with type `cleaned_data`. |
| `clean_sample.csv:reference` | Fixed reference version for distribution checks. |
| `trainval_data.csv` | Training and validation data. |
| `test_data.csv` | Held-out test data. |
| `model_export` | Exported MLflow scikit-learn model, uploaded with type `model_export`. |
| `model_export:prod` | Model selected for test-set evaluation. |

These are the artifact names used by this implementation. The lesson uses `clean_sample.csv`; some rubric wording refers to `clean_data.csv`. The model artifact's name and type is `model_export`.

## Model and results

The inference pipeline combines a column preprocessor with a RandomForestRegressor. Preprocessing includes missing-value imputation, categorical encoding, review-date transformation, and TF-IDF features from listing names. The complete fitted pipeline is exported as an MLflow scikit-learn model.

The data split reserves 20% for testing and 20% of the remaining data for validation. The random seed is 42.

| max_depth | n_estimators | Validation MAE |
| --- | --- | --- |
| 15 | 100 | **34.1276** |
| 10 | 100 | 34.4319 |
| 10 | 200 | 34.4046 |
| 50 | 100 | 34.1963 |
| 50 | 200 | 34.1843 |

The original configuration performed better than the four search combinations, so it was selected for production. The defaults in `config.yaml` retain `max_depth=15` and `n_estimators=100`.

The selected model's held-out test MAE was **33.8477**, with **R² = 0.5641**. MAE is the mean absolute prediction error in dollars per night. Test MAE was comparable to validation MAE, providing no indication of a large generalization gap in this evaluation.

In W&B, open the model artifact's Lineage/Graph view to inspect its relationship to the training data and upstream pipeline steps.

## Releases and new data

- **1.0.0:** Initial pipeline release with the selected model hyperparameters. Running it with `sample2.csv` failed the geographic boundary test: one record lay outside the accepted area. The other five data checks passed.
- **1.0.1:** Adds geographic filtering in `src/basic_cleaning/run.py` before saving the cleaned data.

Run the updated release on the new sample:

```bash
mlflow run https://github.com/SteffanyMcH2os/Project-Build-an-ML-Pipeline-Starter.git -v 1.0.1 -P hydra_options="etl.sample='sample2.csv'"
```

This command runs the tagged GitHub release, rather than uncommitted local edits. Confirm that all data checks and the final training run succeed before submission.

## Possible future improvements

- Use cross-validation to assess how stable model performance is across different splits.
- Explore more TF-IDF features and a broader hyperparameter search.
- Compare the random forest with alternative regression models.
- Monitor distribution changes and prediction quality as new weekly samples arrive.
- Investigate extreme values and missing listing information before introducing additional cleaning rules.

## Acknowledgments

Based on the [Udacity ML pipeline starter kit](https://github.com/udacity/Project-Build-an-ML-Pipeline-Starter)


[License](LICENSE.txt)
