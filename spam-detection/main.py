from linecache import clearcache

from src.ClassificationMetrics import *
from src.data import load_data
from src.train import *
from src.model import LogisticsRegression
from src.vectorizer import BagOfWordsVectorized
def main():
    spam_data = load_data()
    #dropna removes row where either label or message is missing
    #subset tells pandas which columns tyo inspect
    cleaned_data  = spam_data.dropna(subset=["label","message"])
    cleaned_data = cleaned_data[
        cleaned_data["message"].str.strip().ne("") #ne means not equal to an empty string or equiavalent != ""
        & cleaned_data["label"].str.strip().ne("")
    ]
    X = cleaned_data["message"]
    #if it is ham then y will be 0 else 1
    y = encode_target(cleaned_data)


    x_train, x_val, x_test, y_train, y_val, y_test = split_data(X, y)

    vectorizer = BagOfWordsVectorized(5000)
    #Learn vocabulary only from training messages
    x_train_vectorized  = vectorizer.fit_transform(x_train)
    x_val_vectorized = vectorizer.transform(x_val)
    x_test_vectorized = vectorizer.transform(x_test)
    logistics_regression = LogisticsRegression()
    logistics_regression.fit(x_train_vectorized, y_train.to_numpy())
    # print("Training matrix", x_train_vectorized.shape)
    # print("number of weights:", logistics_regression.weights.shape)
    # print("number of bias:", logistics_regression.bias)
    # print("Initial cost:",logistics_regression.cost_history[0])
    # print("Final cost:",logistics_regression.cost_history[-1])

    # print("Vocabulary size:", len(vectorizer.vocabulary))
    # print("First words:", vectorizer.vocabulary[:20])

    #Validation
    validation_probabilities = logistics_regression.predict_probability(x_val_vectorized)
    best_threshold, best_metrics, threshold_results = (
        find_best_threshold(
            y_val.to_numpy(),
            validation_probabilities,
        )
    )
    for result in threshold_results:
        print(
            f"\nThreshold: {result['threshold']:.2f}"
        )
        print(
            f"Precision: {result['precision']:.2%}"
        )
        print(
            f"Recall:    {result['recall']:.2%}"
        )
        print(
            f"F1:        {result['f1']:.2%}"
        )
    validation_predictions = logistics_regression.predict(x_val_vectorized)

    print("\nBest validation threshold")
    print("-------------------------")
    print(f"Threshold: {best_threshold:.2f}")
    print(f"Accuracy:  {best_metrics['accuracy']:.2%}")
    print(f"Precision: {best_metrics['precision']:.2%}")
    print(f"Recall:    {best_metrics['recall']:.2%}")
    print(f"F1:        {best_metrics['f1']:.2%}")

    # Final evaluation: use the threshold selected on validation data.
    test_predictions = logistics_regression.predict(
        x_test_vectorized,
        threshold=best_threshold,
    )
    test_metrics = classification_metrics(
        y_test.to_numpy(),
        test_predictions,
    )

    save_metrics(test_metrics)
    save_predictions(y_test.to_numpy(), test_predictions)

    print("\nFinal test metrics")
    print("------------------")
    print(f"Threshold:       {best_threshold:.2f}")
    print(f"True positives:  {test_metrics['true_positive']}")
    print(f"True negatives:  {test_metrics['true_negative']}")
    print(f"False positives: {test_metrics['false_positive']}")
    print(f"False negatives: {test_metrics['false_negative']}")
    print(f"Accuracy:        {test_metrics['accuracy']:.2%}")
    print(f"Precision:       {test_metrics['precision']:.2%}")
    print(f"Recall:          {test_metrics['recall']:.2%}")
    print(f"F1 score:        {test_metrics['f1']:.2%}")
    # print("Probabilities")
    # print(validation_probabilities[:10])
    # print("Predictions:")
    # print(validation_predictions[:10])
    # print("Actual:")
    # print(y_val.iloc[:10].to_numpy())

    metrics = classification_metrics(y_val, validation_predictions)
    # print("\nValidation metrics")
    # print("------------------")
    # print(f"True positives:  {metrics['true_positive']}")
    # print(f"True negatives:  {metrics['true_negative']}")
    # print(f"False positives: {metrics['false_positive']}")
    # print(f"False negatives: {metrics['false_negative']}")
    # print(f"Accuracy:        {metrics['accuracy']:.2%}")
    # print(f"Precision:       {metrics['precision']:.2%}")
    # print(f"Recall:          {metrics['recall']:.2%}")
    # print(f"F1 score:        {metrics['f1']:.2%}")



if __name__ == "__main__":
    main()
