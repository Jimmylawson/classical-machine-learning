from src.data import load_data,clean_data

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
    rg = np.random.default_rng()
    idx = rg.permutation(X.shape[0])




if __name__ == "__main__":
    main()
