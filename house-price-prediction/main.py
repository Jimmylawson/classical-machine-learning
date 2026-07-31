from src.data import load_data,clean_data


from src.train import split_data
from src.model import  LinearRegression
import matplotlib.pyplot as plt

import numpy as np
def main():
    housing_data = load_data()
    cleaned_data = clean_data(housing_data)
    # print(cleaned_data.head())
    # print(cleaned_data.shape)
    X= cleaned_data.drop(columns=["MedHouseVal"])
    y= cleaned_data["MedHouseVal"]
    print(X.shape)
    print(y.shape)

    #shuffling the data
    # rng = np.random.default_rng(42)
    # idx = rng.permutation(X.shape[0])
    #
    # X_shuffled = X.iloc[idx].reset_index(drop=True)
    # y_shuffled = y.iloc[idx].reset_index(drop=True)
    #
    # m = X_shuffled.shape[0]
    # m_train = int(m * 0.8)
    #
    # X_train = X_shuffled.iloc[:m_train]
    # y_train = y_shuffled.iloc[:m_train]
    #
    # X_test = X_shuffled.iloc[m_train:]
    # y_test = y_shuffled.iloc[m_train:]

    #alternative to shuffle the data and assign them

    X_train,X_val,X_test,y_train,y_val,y_test = split_data(X,y)

    #using mu and standard deviation to reduce to make large value smaller for easier optimization and gradient
    mu  = X_train.mean(axis=0) #axis means compute down the rows for each column
    sigma = X_train.std(axis=0)   # sigma stores the std for each  feature
    sigma[sigma == 0] = 1 #to avoid division by zero
#THIS APPROACH IS STANDARDIZATION
# standardized value = (original value − mean) / standard deviation
#after doing this mean will be 0  and std
    #later i can use Standard Scalar from sklearn
    X_train = (X_train - mu) /sigma
    X_val = (X_val - mu) /sigma
    X_test = (X_test - mu) /sigma

    #resetting index
    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)

    X_val = X_val.reset_index(drop=True)
    y_val = y_val.reset_index(drop=True)

    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    # print("X_train:\n", X_train.head())
    # print("X_val:\n", X_val.head())
    # print("X_test:\n", X_test.head())
    # print("y_train:\n", y_train.head())
    # print("y_val:\n", y_val.head())
    # print("y_test:\n", y_test.head())
    model = LinearRegression()
    model.fit(X_train, y_train)

    # print(model is returned_model)
    # print(returned_model)
    predictions = model.predict(X_val)

    # print("Weights:", model.weights)
    # print("Bias:", model.bias)
    # print("Initial cost:", model.cost_history[0])
    # print("Final cost:", model.cost_history[-1])
    # print("Predictions:")
    # print(predictions[:5])
    # print("Actual values:")
    # print(y_val.iloc[:5].to_numpy())

    from src.evaluate import calculate_metrics
    from src.evaluate import save_metrics
    from src.evaluate import save_predictions
    validation_r2 = model.score(X_val, y_val)
    print("Validation R²:", validation_r2)

    #final evaluation
    test_predictions = np.asarray(model.predict(X_test))
    test_actual = y_test.to_numpy()

    test_errors = test_predictions - test_actual

    test_mse = np.mean(test_errors ** 2)
    test_rmse = np.sqrt(test_mse)
    test_mae = np.mean(np.abs(test_errors))
    test_r2 = model.score(X_test, y_test)
    #evaluate 
    test_metrics = calculate_metrics(
        test_actual,
        test_predictions,
    )

    all_metrics = {
        "model": "linear_regression_from_scratch",
        "learning_rate": model.learning_rate,
        "iterations": model.iterations,
        "validation_r2": float(validation_r2),
        "test": test_metrics,
    }

    save_metrics(all_metrics)
    save_predictions(test_actual, test_predictions)

    print("Results saved successfully.")

    print(f"Test MAE:  {test_mae:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"Test R²:   {test_r2:.4f}")

    print(f"Approximate MAE:  ${test_mae * 100_000:,.0f}")
    print(f"Approximate RMSE: ${test_rmse * 100_000:,.0f}")











if __name__ == "__main__":
    main()
