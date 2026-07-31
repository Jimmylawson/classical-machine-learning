from sklearn.model_selection import train_test_split


def split_data(X,y):
    X_train,X_temp,y_train,y_temp = train_test_split(
    X,
    y,
    test_size=0.4,
    random_state=42,
    shuffle=True,
    stratify=y
)


    X_val,X_test,y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        random_state=42,
        shuffle=True,
        stratify=y_temp
    )

    return X_train, X_val, X_test, y_train, y_val, y_test

def encode_target(data):
    labels = data["label"].str.strip().str.lower()
    y= labels.map({
        "ham": 0,
        "spam": 1
    })

    if y.isna().any():
        unexpected = labels[y.isna()].unique()
        raise ValueError(f"Unexpected labels found: {unexpected}")
    return y.astype(int)