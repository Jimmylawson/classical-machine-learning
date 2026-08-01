import numpy as np


def classification_metrics(actual,prediction):
    actual = np.array(actual)
    prediction = np.array(prediction)

    true_positive = np.sum(
        (actual == 1) & (prediction == 1)
    )
    true_negative = np.sum(
        (actual == 0) & (prediction  == 0)
    )

    false_positive = np.sum(
        (actual == 0) & (prediction  == 1)
    )
    false_negative = np.sum(
        (actual == 1) & (prediction  == 0)
    )

    total = len(actual)
    accuracy  = (true_positive + true_negative) / total
    precision_denominator = true_positive + false_positive

    precision = (
        true_positive / precision_denominator
        if precision_denominator > 0
        else 0.0
    )
    recall_denominator = (
            true_positive + false_negative
    )

    recall = (
        true_positive / recall_denominator
        if recall_denominator > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    return{
        "true_positive" :int(true_positive),
        "true_negative":int(true_negative),
        "false_positive":int(false_positive),
        "false_negative":int(false_negative),
        "accuracy":accuracy,
        "precision":precision,
        "recall":recall,
        "f1":f1

    }


