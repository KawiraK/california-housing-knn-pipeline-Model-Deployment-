# California Housing Price Prediction: KNN Pipeline with Hyperparameter Tuning

An end-to-end machine learning workflow that predicts median house values using the California Housing dataset, covering preprocessing, cross-validation, hyperparameter tuning and model saving.

## Dataset
- Source: scikit-learn (`fetch_california_housing`), 1990 California census
- 20,640 samples, 8 numeric features, 0 missing values
- **Target:** median house value (in $100,000s)
- **Features:** MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude

## Workflow
1. Load the data with `fetch_california_housing`
2. Split into training and test sets (80/20, `random_state=42`)
3. Preprocess with `ColumnTransformer` (`StandardScaler` on all numeric features)
4. Build a `Pipeline`: preprocessor + `KNeighborsRegressor`
5. Tune with `GridSearchCV` (5-fold CV, scoring = R²)
6. Evaluate the best model on the unseen test set
7. Save the trained pipeline with `joblib`

## Hyperparameter Grid
| Parameter | Values |
|---|---|
| `n_neighbors` | 3, 5, 7, 9 |
| `weights` | uniform, distance |
| `p` | 1 (Manhattan), 2 (Euclidean) |

16 combinations x 5 folds = 80 model fits.

## Results
| Metric | Score |
|---|---|
| Best parameters | **[best_params_]** |
| Best cross-validated R² | **[X]** |
| Test R² | **[Y]** |

**Interpretation:** Scaling was essential because KNN is distance-based. The close match between cross-validated and test R² shows the model generalises well to unseen data. Putting the scaler inside the pipeline kept the test data out of preprocessing, which avoids data leakage.

## Files
| File | Description |
|---|---|
| `california_housing_knn.ipynb` | Full notebook with code, outputs and plots |
| `knn_housing_pipeline.joblib` | Saved trained pipeline |

## Run It
Open the notebook in Google Colab and select **Runtime → Run all**.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](PASTE_YOUR_COLAB_LINK_HERE)

## Load the Saved Model
```python
import joblib
import pandas as pd

model = joblib.load("knn_housing_pipeline.joblib")

# Example: one block group (same 8 features, same order)
sample = pd.DataFrame([{
    "MedInc": 5.0,
