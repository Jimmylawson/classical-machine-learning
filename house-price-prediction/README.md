# California House-Price Prediction From Scratch

This project implements multivariable linear regression from scratch with NumPy. It was built while studying the CS229 concepts of training, validation, testing, feature standardization, gradient descent, generalization, and regression evaluation.

The regression model does not use `sklearn.linear_model.LinearRegression`. Scikit-learn is used only to download the California Housing dataset and create reproducible data splits.

## Project goals

- Understand how a supervised regression project is organized.
- Keep raw data, source code, results, and models separate.
- Implement the linear hypothesis, cost function, gradients, and batch gradient descent.
- Prevent data leakage during feature standardization.
- Use training, validation, and test sets for their proper purposes.
- Evaluate predictions with MAE, MSE, RMSE, and R².
- Save summary metrics and row-level test predictions.

## Project structure

```text
house-price-prediction/
├── data/
│   └── raw/
│       └── california_housing.csv
├── models/
│   └── linear_regression.pkl   # Optional model artifact
├── notebooks/
│   └── exploration.ipynb
├── results/
│   ├── metrics.json
│   └── prediction.csv
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── evaluate.py
│   ├── model.py
│   ├── train.py
│   └── utils.py
├── main.py
├── README.md
└── requirements.txt
```

### File responsibilities

- `src/data.py`: downloads, loads, and cleans the dataset.
- `src/train.py`: creates the training, validation, and test splits.
- `src/model.py`: contains the from-scratch linear-regression model.
- `src/evaluate.py`: calculates metrics and saves results.
- `main.py`: coordinates the complete workflow.
- `results/metrics.json`: stores model settings and final metrics.
- `results/prediction.csv`: stores actual values, predictions, and errors for each test row.

## Dataset

The project uses scikit-learn's California Housing dataset. It contains 20,640 observations, eight numeric features, and one regression target.

The features are:

1. `MedInc` — median income
2. `HouseAge` — median house age
3. `AveRooms` — average rooms
4. `AveBedrms` — average bedrooms
5. `Population` — block-group population
6. `AveOccup` — average occupancy
7. `Latitude`
8. `Longitude`

The target is `MedHouseVal`. It is expressed in units of $100,000, so a target value of `2.5` represents approximately $250,000.

When `data/raw/california_housing.csv` is missing, `src/data.py` downloads it automatically. The raw file is kept unchanged. Basic cleaning removes duplicate rows and rows with a missing target.

## Data preparation

The features and target are separated as:

```python
X = data.drop(columns=["MedHouseVal"])
y = data["MedHouseVal"]
```

The dataset uses a reproducible 60/20/20 split:

- 60% training data
- 20% validation data
- 20% test data

The first split places 40% in a temporary set. That temporary set is divided equally to produce the validation and test sets. A random state of `42` ensures that the same rows are selected each time.

## Feature standardization

Each feature is standardized with statistics calculated from the training set only:

```text
standardized value = (value - training mean) / training standard deviation
```

The same training mean and standard deviation are applied to validation and test data. This prevents information from those unseen sets from leaking into model training.

Features with a standard deviation of zero use a divisor of one to avoid division by zero. Such a constant feature becomes zero after standardization.

## Linear-regression model

The prediction function is:

```text
y_hat = Xw + b
```

Here, `w` contains one learned weight per feature and `b` is the learned bias. The same parameters are used for every observation.

The model minimizes the squared-error cost:

```text
J(w, b) = (1 / 2m) * sum((y_hat - y)^2)
```

For batch gradient descent, every iteration uses all training observations. The gradients are:

```text
dw = (1 / m) * X.T @ (y_hat - y)
db = (1 / m) * sum(y_hat - y)
```

The parameters are updated with:

```text
w = w - learning_rate * dw
b = b - learning_rate * db
```

This project uses:

```text
learning rate = 0.01
iterations    = 1000
```

The cost from every iteration is stored in `cost_history` so convergence can be inspected later.

## Evaluation

The validation set is used to examine model performance while developing the model. The test set is reserved for the final evaluation.

The project reports four regression metrics:

- **MAE (Mean Absolute Error):** the average absolute prediction error. Lower is better.
- **MSE (Mean Squared Error):** the average squared prediction error. It penalizes large errors more heavily. Lower is better.
- **RMSE (Root Mean Squared Error):** the square root of MSE, expressed in the target's original units. Lower is better.
- **R² (Coefficient of Determination):** the proportion of target variation explained by the model. Higher is better; `1` is perfect, `0` matches a mean-prediction baseline, and a negative value is worse than that baseline.

R² is not classification accuracy.

## Results

The final run produced:

| Metric | Result |
|---|---:|
| Validation R² | 0.5721 |
| Test MAE | 0.5470 |
| Test MSE | 0.5645 |
| Test RMSE | 0.7513 |
| Test R² | 0.5883 |
| Approximate test MAE | $54,702 |
| Approximate test RMSE | $75,132 |

The validation and test R² values are close, suggesting consistent generalization without an obvious validation-to-test performance collapse. The test R² indicates that this baseline linear model explains approximately 58.8% of the variation in unseen target values.

## Setup

From the project directory, create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

The main dependencies are:

- NumPy for numerical operations and the from-scratch algorithm
- pandas for CSV and DataFrame operations
- scikit-learn for the dataset loader and data splitting
- Matplotlib for optional plotting

## Run the project

From `house-price-prediction/`, run:

```bash
python main.py
```

The program will:

1. Download the dataset if it is missing.
2. Load and clean the data.
3. Separate features and target.
4. Create the train, validation, and test splits.
5. Standardize features using training statistics.
6. Train linear regression with batch gradient descent.
7. Evaluate validation and test performance.
8. Write `results/metrics.json` and `results/prediction.csv`.

## Result files

`metrics.json` contains the model name, learning rate, iteration count, validation R², and final test metrics.

`prediction.csv` contains:

- the actual target
- the predicted target
- the signed prediction error
- the absolute error
- the approximate actual dollar value
- the approximate predicted dollar value

## Possible extensions

- Plot cost versus gradient-descent iterations.
- Plot actual versus predicted validation values.
- Implement L2/Ridge regularization from scratch.
- Compare training and validation errors to diagnose bias and variance.
- Implement early stopping based on cost convergence.
- Save the trained model and preprocessing statistics for reuse.
- Add automated tests for the model and metric functions.

## Key lessons

- A model should learn only from training data.
- Validation data supports model and hyperparameter choices.
- Test data provides the final estimate of generalization.
- Standardization statistics must come only from training data.
- Batch gradient descent uses every training row for each parameter update.
- MAE, RMSE, and R² describe different aspects of regression performance.
- A reproducible project separates data, source code, results, and documentation.
