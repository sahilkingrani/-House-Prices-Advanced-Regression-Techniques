# House Prices: Advanced Regression Techniques

End-to-end regression pipeline for the Kaggle competition [House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques): data-aware missing value handling, feature engineering, leakage-safe validation, and a Ridge + XGBoost blend.

**Final result:** 5-fold CV RMSE **0.1113** (log price) | Kaggle leaderboard score **0.12857**

## Problem

Predict the sale price of houses in Ames, Iowa from 79 features describing the property (size, quality, neighborhood, garage, basement, and more).

- **Task:** regression
- **Metric:** RMSE between the log of predicted and the log of actual `SalePrice` (lower is better)
- **Data:** 1,460 training houses, 1,459 test houses

> The competition data is not included in this repository, as per the competition rules. Download it from the [competition page](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data).

## Results

| Measure | RMSE (log price) |
|---|---|
| Dummy baseline (always predicts the mean) | 0.3993 |
| Ridge, tuned (alpha=300), 5-fold CV | 0.1190 |
| XGBoost, 5-fold CV | 0.1147 |
| **Blend: 50% Ridge + 50% XGBoost, 5-fold CV** | **0.1113** |
| **Kaggle leaderboard** | **0.12857** |

The leaderboard score is higher than the CV score. See "Honest notes" below for the likely reasons.

## Approach

### 1. Exploratory Data Analysis

- **Target is right-skewed** (skewness 1.88). After `log1p` it is almost symmetric (0.12), so models are trained on log price. Predictions are converted back with `expm1` before submission.
- **Most NaN values are not missing data.** For columns like `PoolQC`, `GarageType`, `BsmtQual` and `FireplaceQu`, the NaN count exactly matched the count of houses without that feature (for example 1,453 NaN in `PoolQC` and 1,453 houses with `PoolArea == 0`). This was verified before any imputation.
- **Strongest signals:** `OverallQual` (correlation 0.817 with log price) and `GrLivArea` (0.701).
- **Neighborhood matters a lot:** median price ranges from 88,000 (MeadowV) to 315,000 (NridgHt), about 3.6x.
- **Quality ratings are ordered:** median price by `ExterQual` is Ex 364,607 > Gd 220,000 > TA 139,450 > Fa 82,250, so ordinal encoding is appropriate.
- **Two outliers:** two very large, top-quality houses sold for ordinary prices (Ids 524 and 1299, both partial sales).

### 2. Preprocessing

- Removed the two outliers (Ids 524 and 1299) from the **training set only**. Test rows are never removed.
- Filled "feature absent" NaN with `'None'` (text columns) or `0` (numeric columns).
- Filled genuinely missing values using statistics learned from the training set only: `LotFrontage` by neighborhood median, `GarageYrBlt` from `YearBuilt`, and the training mode for a few categorical columns.
- Fixed a clear typo in the test set (`GarageYrBlt` = 2207 changed to 2007) and clipped two negative house ages to 0 (partial sales recorded before completion).
- Dropped `Utilities` (1,457 of 1,458 training rows have the same value).
- Ordinal-encoded 14 quality and rating columns; kept `MSSubClass` as a category because it is a code, not a number.
- One-hot encoded the remaining categorical columns (259 features in total).

### 3. Feature Engineering

| Feature | Definition | Correlation with log price |
|---|---|---|
| `TotalSF` | basement + 1st floor + 2nd floor area | 0.83 |
| `TotalBath` | full baths + 0.5 x half baths (including basement) | 0.68 |
| `HouseAge` | year sold minus year built | -0.59 |
| `RemodAge` | year sold minus year remodeled | -0.57 |
| `HasPool` | pool area greater than 0 | 0.08 |

### 4. Validation

- 5-fold `KFold` with shuffling (regression target, independent houses, so no stratification or grouping needed).
- Scaling is done inside a scikit-learn `Pipeline`, so each fold learns its scaler only from its own training part.
- A dummy model (0.3993) set the floor that every real model had to beat.

### 5. Modeling

Changes were made one at a time and logged.

## Experiment Log

| # | Model | CV RMSE (log) | Notes |
|---|---|---|---|
| 0 | Dummy (mean) | 0.3993 | Floor |
| 1 | Ridge (alpha=1) | 0.1310 | Baseline |
| 2 | Ridge (alpha=300) | 0.1190 | Tuned; better in 5 of 5 folds |
| 3 | Ridge + log of skewed features | 0.1185 | Gain within noise, not adopted |
| 4 | XGBoost (depth 3, lr 0.05, 500 trees) | 0.1147 | Different errors from Ridge |
| 5 | **Blend 50/50 Ridge + XGBoost** | **0.1113** | Errors correlated at 0.81, so blending helped |

## Model Interpretation

- Both models rely most on quality and size: `OverallQual` and `TotalSF` are near the top for Ridge and XGBoost.
- For XGBoost, three features (`OverallQual`, `ExterQual`, `TotalSF`) account for about 38% of the importance. The engineered `TotalSF` ranks third.
- Feature importance and coefficients show associations, not causes, and correlated features (area columns, garage columns) share credit between them.
- **Largest errors:** 8 houses (0.55% of the data) produced about 18% of the total squared error. Six of the eight were not normal sales (foreclosure-type, family, or partial sales), so their prices do not reflect market value.

## Honest Notes

- **CV vs leaderboard gap (0.1113 vs 0.12857).** Likely contributors: (1) picking the best of many configurations on the same CV folds makes CV optimistic, (2) a few unusual sales dominate RMSE, and (3) preprocessing statistics such as the `LotFrontage` medians were computed on the full training set before CV. I cannot say how much each contributes.
- **One test house had features far outside the training range** (living area and basement area above the training maximum), and Ridge extrapolated it to a very high price. As a systematic safety step, all test features were clipped to the training min/max range (this changed 22 houses). No individual predictions were edited by hand.
- **No leaderboard-based tuning.** I did not submit multiple variants to pick the best one on the test set.

## What I Would Improve Next

- A robust loss (for example Huber) or sample weighting for abnormal sales.
- Lasso / ElasticNet, LightGBM, CatBoost, and stacking instead of a fixed 50/50 blend.
- Nested cross-validation to measure the selection bias directly.
- Compute all imputation statistics inside each CV fold.

## Skills Demonstrated

Exploratory data analysis, data-aware missing value handling, target transformation, ordinal and one-hot encoding, feature engineering, leakage-safe validation with pipelines, Ridge and XGBoost regression, model blending, error analysis, and honest comparison of validation and leaderboard scores.

## Tech Stack

Python, pandas, NumPy, scikit-learn, XGBoost, matplotlib, Kaggle Notebooks.

## How to Run

1. Download the data from the [competition page](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data).
2. Open the notebook (`.ipynb`) in this repository in Kaggle or Jupyter.
3. Update the data paths if needed and run all cells from top to bottom.

---
