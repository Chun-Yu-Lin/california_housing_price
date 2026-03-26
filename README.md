# California Housing Price Prediction

This project explores the California Housing dataset end-to-end, with a specific focus on diagnosing multicollinearity and comparing model pipelines under the same preprocessing setup.

## Project Goal

Predict `median_house_value` and study how feature collinearity affects:
- unregularized linear regression,
- regularized linear regression,
- tree-based models.

## Dataset

- Source: California Housing dataset.
- Target: `median_house_value`.
- Split strategy: stratified train/test split using income categories (engineered from `median_income`).

## Notebook Summary

The notebook `california_housing_price.ipynb` follows this workflow:

1. **Load and inspect data**
   - Download/load dataset.
   - Review schema, missing values, and basic distributions.

2. **Preliminary EDA**
   - Numeric and non-numeric feature exploration.
   - Distribution checks and category inspection.

3. **Train/test split**
   - Stratified split by income bins to preserve distribution.

4. **In-depth EDA on train set**
   - Geospatial visualization on California map.
   - Correlation analysis with target.

5. **Data preparation**
   - Median imputation for missing `total_bedrooms`.
   - VIF diagnostics to quantify multicollinearity.
   - Feature engineering with ratio features:
     - `rooms_per_household`
     - `bedrooms_per_room`
     - `population_per_household`
   - Categorical one-hot encoding (`ocean_proximity`).
   - Cluster-based location features from `longitude` and `latitude`.
   - Numeric standardization for linear models.

6. **Pipeline comparison**
   - `linreg`: OLS linear regression with shared preprocessor.
   - `linreg_ratio`: OLS with ratio-focused feature set.
   - `ridge`: regularized linear model (Ridge).
   - `rf`: random forest regressor.

7. **Randomized hyperparameter search**
   - `RandomizedSearchCV` applied to all four pipelines with model-specific parameter spaces.

## Key Findings

### Multicollinearity diagnostics

VIF analysis confirms strong collinearity among household/room/population-related variables, motivating regularization and feature engineering.

### Baseline pipeline comparison (single train/test split)

From the notebook output table:
<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>model</th>
      <th>train_rmse</th>
      <th>test_rmse</th>
      <th>train_r2</th>
      <th>test_r2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>linreg</td>
      <td>67270.964535</td>
      <td>70457.864916</td>
      <td>0.659657</td>
      <td>0.629141</td>
    </tr>
    <tr>
      <th>1</th>
      <td>linreg_ratio</td>
      <td>70567.288435</td>
      <td>72237.380125</td>
      <td>0.625485</td>
      <td>0.610171</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ridge</td>
      <td>67275.636727</td>
      <td>70382.028000</td>
      <td>0.659609</td>
      <td>0.629938</td>
    </tr>
    <tr>
      <th>3</th>
      <td>rf</td>
      <td>21539.109010</td>
      <td>59371.132412</td>
      <td>0.965109</td>
      <td>0.736670</td>
    </tr>
  </tbody>
</table>
</div>

Interpretation:
- Ridge and OLS are very close on this setup, with Ridge slightly better on test data.
- The ratio-only linear variant somehow underperforms the raw linear baseline.
- Random forest gives the best predictive performance on the holdout test split.

### Random search outcomes

- Random search preserves the same ranking trend: `rf` best, then `ridge`/`linreg`, then `linreg_ratio`.
- Some Ridge trials fail because `lbfgs` is only valid when `positive=True` in scikit-learn, which appears in the notebook warnings.
- The `test_rmse` column in the random-search results is currently computed using `mean_squared_error(...)` without square root, so those values are actually MSE-scale despite the column name.

## Conclusions

1. **Multicollinearity is present and material** in the dataset, especially among room/household/population features.
2. **Regularization helps stabilize linear modeling**, but does not dramatically outperform OLS in this exact feature setup.
3. **Tree-based models are strongest for this problem** due to non-linear and interaction-heavy structure in geospatial and socioeconomic features.