from src.data import load_data
from src.train import *

def main():
    spam_data = load_data()
    print(spam_data.head())
    print(spam_data.shape)
    X = spam_data.dropna(subset=["label", "message"])

    #if it is ham then y will be 0 else 1
    y = encode_target(X)

    x_train, x_val, x_test, y_train, y_val, y_test = split_data(X, y)
    # print("Training set shape:", x_train.shape)
    # print("Validation set shape:", x_val.shape)
    # print("Test set shape:", x_test.shape)

if __name__ == "__main__":
    main()