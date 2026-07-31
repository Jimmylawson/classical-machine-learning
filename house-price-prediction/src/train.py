from sklearn.model_selection import train_test_split


def split_data(X,y):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.4,
        random_state=42,
        shuffle=True,
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5, # meaning use half of the test_size which is 40%
        random_state=42,
        shuffle=True,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test