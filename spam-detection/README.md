# SMS Spam Detection from Scratch

This project classifies SMS messages as **ham** (legitimate) or **spam**. It was built as a learning project while studying classical machine learning and CS229 concepts.

The main machine-learning components were implemented from scratch with NumPy:

- Text tokenization
- Binary bag-of-words vectorization
- Logistic regression
- Binary cross-entropy cost
- Batch gradient descent
- Classification metrics
- Validation-threshold search

Scikit-learn is used only to split the data into stratified training, validation, and test sets. It is not used to vectorize the messages, train the model, generate predictions, or calculate the final metrics.

## Dataset

The project uses the **SMS Spam Collection** dataset. The raw file contains two tab-separated fields on each row:

1. `label` — `ham` or `spam`
2. `message` — the SMS text

The raw dataset is stored at:

```text
data/raw/SMSSpamCollection
```

When the project runs, pandas reads the tab-separated file and saves a CSV copy at:

```text
data/processed/spam_detection.csv
```

The labels are converted into numbers:

```text
ham  -> 0
spam -> 1
```

Rows with missing or empty labels or messages are removed before training.

## Project Structure

```text
spam-detection/
├── data/
│   ├── raw/
│   │   └── SMSSpamCollection
│   └── processed/
│       └── spam_detection.csv
├── results/
│   ├── metrics.json
│   └── spam_predictions.csv
├── src/
│   ├── ClassificationMetrics.py
│   ├── data.py
│   ├── evaluate.py
│   ├── model.py
│   ├── train.py
│   └── vectorizer.py
├── main.py
└── README.md
```

### File responsibilities

- `main.py` coordinates the complete training and evaluation pipeline.
- `src/data.py` loads the raw dataset and saves the processed CSV.
- `src/train.py` encodes the target and creates the data splits.
- `src/vectorizer.py` tokenizes messages and converts them into numeric vectors.
- `src/model.py` contains the logistic regression implementation.
- `src/ClassificationMetrics.py` calculates metrics, selects a threshold, and saves results.
- `results/metrics.json` contains the final test metrics.
- `results/spam_predictions.csv` contains the actual and predicted test labels.

## Data Splitting

The cleaned dataset is divided into:

- **60% training data** — learns the vocabulary, weights, and bias.
- **20% validation data** — selects the classification threshold.
- **20% test data** — provides the final evaluation.

The splits use `random_state=42` so the same split can be reproduced. They are also stratified, which keeps approximately the same ham-to-spam ratio in each set.

The test set is used only after training and threshold selection are complete. It is not used to tune the model.

## Text Vectorization

Logistic regression cannot train directly on text, so every message is converted into a numeric feature vector.

The custom vectorizer performs the following steps:

1. Converts the message to lowercase.
2. Extracts words with a regular expression.
3. Counts how many training messages contain each word.
4. Keeps the 5,000 most common words.
5. Assigns an index to every vocabulary word.
6. Creates a binary vector for each message.

For a particular word feature:

```text
0 = the word is absent from the message
1 = the word is present in the message
```

The vocabulary is learned only from `x_train`. The validation and test messages are transformed using that same training vocabulary. This prevents data leakage.

The resulting training matrix has one row per training message and one column per vocabulary word:

```text
(3343, 5000)
```

## Logistic Regression

For a message vector `X`, the model first calculates a linear score:

```text
z = Xw + b
```

The sigmoid function converts that score into a probability between 0 and 1:

```text
probability = 1 / (1 + exp(-z))
```

The model learns one weight for each of the 5,000 word features and one bias value.

### Cost function

Training uses binary cross-entropy, also called negative log-likelihood:

```text
cost = -(1/m) * sum(y*log(p) + (1-y)*log(1-p))
```

The probabilities are clipped before taking the logarithm to avoid `log(0)`.

### Gradient descent

For every training iteration, the model:

1. Calculates probabilities.
2. Calculates the prediction errors.
3. Calculates the gradients for the weights and bias.
4. Updates the weights and bias.
5. Records the new cost in `cost_history`.

The model currently uses:

```text
learning rate = 0.01
iterations    = 1000
```

## Classification Threshold

Logistic regression returns probabilities, but the final output must be either ham or spam. A threshold converts the probability into a class:

```text
probability >= threshold -> spam (1)
probability < threshold  -> ham  (0)
```

The default threshold of `0.50` produced very high precision but extremely low recall, meaning the model almost never predicted spam.

Thresholds from `0.05` through `0.50` were compared using the validation set. The threshold with the highest validation F1 score was:

```text
best threshold = 0.25
```

This threshold was then frozen and used once on the test set. It should not be changed based on the test results.

## Final Test Results

The final evaluation at threshold `0.25` produced:

| Metric | Result |
|---|---:|
| Accuracy | 96.23% |
| Precision | 88.57% |
| Recall | 82.67% |
| F1 score | 85.52% |
| True positives | 124 |
| True negatives | 949 |
| False positives | 16 |
| False negatives | 26 |

### Interpretation

- **Accuracy:** 96.23% of all test messages were classified correctly.
- **Precision:** Of the messages predicted as spam, 88.57% were actually spam.
- **Recall:** Of all messages that were actually spam, the model detected 82.67%.
- **F1 score:** 85.52% represents the balance between precision and recall.
- **False positives:** 16 legitimate messages were incorrectly classified as spam.
- **False negatives:** 26 spam messages were incorrectly classified as ham.

Because spam is the minority class, accuracy alone is not enough to judge this model. Precision, recall, and F1 provide a more useful picture of spam-detection performance.

## Saved Results

`results/metrics.json` stores the final test metrics in JSON format:

```json
{
    "true_positive": 124,
    "true_negative": 949,
    "false_positive": 16,
    "false_negative": 26,
    "accuracy": 0.9623318385650225,
    "precision": 0.8857142857142857,
    "recall": 0.8266666666666667,
    "f1": 0.8551724137931035
}
```

`results/spam_predictions.csv` stores one row for every test example:

```csv
actual,prediction
0,0
1,1
1,0
```

## Installation

The shared virtual environment and `requirements.txt` are located in the parent `classical ML` directory.

From the parent directory, create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Running the Project

Move into the project directory and run `main.py`:

```bash
cd spam-detection
python main.py
```

The program will:

1. Load and clean the messages.
2. Create stratified training, validation, and test sets.
3. Build the vocabulary from the training messages.
4. Vectorize all three sets.
5. Train logistic regression from scratch.
6. Select the best threshold using validation F1.
7. Evaluate the frozen model and threshold on the test set.
8. Save the test metrics and predictions in `results/`.

## What I Learned

This project demonstrates:

- How raw text becomes numeric machine-learning features
- Why a vocabulary must be learned only from training data
- How logistic regression produces probabilities
- How binary cross-entropy measures classification error
- How gradient descent learns weights and bias
- Why imbalanced classification requires more than accuracy
- The difference between precision and recall
- How a classification threshold changes model behavior
- Why validation data is used for tuning and test data is used for final evaluation

## Possible Future Improvements

- Save predicted probabilities alongside the final classes.
- Add L2 regularization to reduce overfitting.
- Compare binary bag-of-words with word-count or TF-IDF features.
- Add n-gram features such as two-word phrases.
- Plot the cost history during training.
- Plot precision, recall, and F1 against the threshold.
- Add automated tests for the tokenizer, vectorizer, model, and metrics.
