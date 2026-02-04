# Student Exam Score Prediction Pipeline (Ensemble Regression)

## Overview

This notebook implements a **regression-based prediction pipeline** using multiple machine learning models and ensembling techniques.  
The goal is to predict a continuous target variable using structured tabular data.

The workflow follows a standard data science process:
- Data loading and inspection
- Feature preparation
- Model training using multiple regressors
- Cross-validation and evaluation
- Prediction and ensembling

---

## Data Loading

The dataset is loaded from CSV files and inspected to understand its structure.

```python
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

print(train.shape)
train.head()
```

---

## Models Used

Multiple regression models are trained and compared:

- **Ridge Regression**
- **XGBoost Regressor**
- **CatBoost Regressor**

```python
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
```

---

## Cross-Validation & Training

K-Fold cross-validation is used to train models and evaluate performance using RMSE.

```python
ridge_oof[va_idx] = ridge.predict(X_va)
print(f"Fold RMSE: {rmse(y_va, ridge_oof[va_idx]):.5f}")
```

```python
print("Ridge CV RMSE:", rmse(y, ridge_oof))
```

---

## Ensembling Predictions

Predictions from different models are combined to improve performance.

```python
final_pred = 0.6 * ridge_pred + 0.4 * cat_pred
```

---

## Evaluation Metric

The primary evaluation metric used is **Root Mean Squared Error (RMSE)**.

```python
RMSE = sqrt(mean((y_true - y_pred)^2))
```

RMSE penalizes large errors more heavily and is well-suited for regression problems.

---


### Ridge Regression
Ridge Regression is a **linear regression model with L2 regularization**. It adds a penalty term to reduce large coefficients and helps prevent overfitting, especially when features are correlated.

```python
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
```

---

### XGBoost Regressor
XGBoost is a **gradient boosting tree-based model** that builds trees sequentially to correct previous errors. It is known for strong performance on structured datasets.

```python
xgb = XGBRegressor(n_estimators=300, learning_rate=0.05)
xgb.fit(X_train, y_train)
```

---

### CatBoost Regressor
CatBoost is a **gradient boosting algorithm** that handles categorical features effectively and reduces target leakage using ordered boosting.

```python
cat = CatBoostRegressor(iterations=300, learning_rate=0.05, verbose=False)
cat.fit(X_train, y_train)
```

---

### Why Ensembling Helps
Each model captures different patterns:
- Ridge captures linear relationships
- XGBoost captures complex non-linear interactions
- CatBoost improves robustness and generalization

Combining predictions improves overall performance and stability.

---


## Requirements

```bash
pip install pandas numpy scikit-learn xgboost catboost
```

---


## Author

**Manasa Vijayendra Gokak**  
Graduate Student – Data Science  

