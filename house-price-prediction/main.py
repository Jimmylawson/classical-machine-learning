from src.data import load_data,clean_data

from sklearn.model_selection import train_test_split
from src.train import split_data

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

#THIS APPROACH IS STANDARDIZATION
# standardized value = (original value − mean) / standard deviation
#after doign this mean will be 0  and std
    X_train = (X_train - mu) /sigma
    X_val = (X_val - mu) /sigma
    X_test = (X_test - mu) /sigma







if __name__ == "__main__":
    main()
