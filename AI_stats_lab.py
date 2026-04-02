"""
AIstats_lab.py

Student starter file for:
1. Naive Bayes spam classification
2. K-Nearest Neighbors on Iris
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def accuracy_score(y_true, y_pred):
    """
    Compute classification accuracy.
    """
    return float(np.mean(y_true == y_pred))


# =========================
# Q1 Naive Bayes
# =========================

def naive_bayes_mle_spam():
    """
    Implement Naive Bayes spam classification using simple MLE.
    """
    texts = [
        "win money now",
        "limited offer win cash",
        "cheap meds available",
        "win big prize now",
        "exclusive offer buy now",
        "cheap pills buy cheap meds",
        "win lottery claim prize",
        "urgent offer win money",
        "free cash bonus now",
        "buy meds online cheap",
        "meeting schedule tomorrow",
        "project discussion meeting",
        "please review the report",
        "team meeting agenda today",
        "project deadline discussion",
        "review the project document",
        "schedule a meeting tomorrow",
        "please send the report",
        "discussion on project update",
        "team sync meeting notes"
    ]

    labels = np.array([
        1,1,1,1,1,1,1,1,1,1,
        0,0,0,0,0,0,0,0,0,0
    ])

    test_email = "win cash prize now"

    # Step 1: Tokenize texts
    tokenized = [text.lower().split() for text in texts]

    # Step 2: Build vocabulary
    vocab = set()
    for tokens in tokenized:
        vocab.update(tokens)
    vocab = sorted(vocab)

    # Step 3: Compute class priors P(spam) and P(not spam)
    n_total = len(labels)
    classes = [0, 1]
    priors = {}
    for c in classes:
        priors[c] = float(np.sum(labels == c) / n_total)

    # Step 4: Compute word probabilities using simple MLE
    # word_probs[c][word] = count(word in class c) / total words in class c
    word_probs = {c: {} for c in classes}

    for c in classes:
        # Collect all words in this class
        class_words = []
        for tokens, label in zip(tokenized, labels):
            if label == c:
                class_words.extend(tokens)

        total_word_count = len(class_words)

        # Count each word
        word_count = {}
        for word in class_words:
            word_count[word] = word_count.get(word, 0) + 1

        # MLE probability: count / total (no smoothing)
        for word in vocab:
            word_probs[c][word] = word_count.get(word, 0) / total_word_count

    # Step 5: Predict class of test_email using log probabilities
    # log P(c | email) ∝ log P(c) + sum of log P(word | c)
    # If a word has prob 0, it zeros out the whole class probability
    test_tokens = test_email.lower().split()

    log_posteriors = {}
    for c in classes:
        log_prob = np.log(priors[c])
        for word in test_tokens:
            if word in word_probs[c]:
                p = word_probs[c][word]
                if p == 0:
                    log_prob = -np.inf  # word never seen in this class
                    break
                log_prob += np.log(p)
            # if word not in vocab at all, skip it
        log_posteriors[c] = log_prob

    # Predict class with highest log posterior
    prediction = max(log_posteriors, key=lambda c: log_posteriors[c])

    return priors, word_probs, prediction


# =========================
# Q2 KNN
# =========================

def knn_iris(k=3, test_size=0.2, seed=0):
    """
    Implement KNN from scratch on the Iris dataset.
    """
    # Step 1: Load iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    # Step 2: Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    # Step 3 & 4: Euclidean distance + KNN prediction
    def predict(X_train, y_train, X_query, k):
        predictions = []
        for query in X_query:
            # Euclidean distances to all training points
            distances = np.sqrt(np.sum((X_train - query) ** 2, axis=1))
            # Indices of k nearest neighbors
            k_nearest_indices = np.argsort(distances)[:k]
            # Labels of k nearest neighbors
            k_nearest_labels = y_train[k_nearest_indices]
            # Majority vote
            counts = np.bincount(k_nearest_labels)
            predictions.append(np.argmax(counts))
        return np.array(predictions)

    # Step 5: Compute train and test predictions
    train_predictions = predict(X_train, y_train, X_train, k)
    test_predictions = predict(X_train, y_train, X_test, k)

    # Step 6: Compute accuracies
    train_accuracy = accuracy_score(y_train, train_predictions)
    test_accuracy = accuracy_score(y_test, test_predictions)

    return train_accuracy, test_accuracy, test_predictions
