#!/usr/bin/env python
"""Chord function training and evaluation."""

import csv
import glob
import numpy as np

from typing import Dict, List, Tuple
from sklearn import feature_extraction
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.model_selection import GridSearchCV


FeatureVector = Dict[str, str]
FeatureVectors = List[FeatureVector]

# Mozart
TRAIN_TSV = "outputFiles/cleanedFiles/cleanedMozartFiles/train/*.tsv"
TEST_TSV = "outputFiles/cleanedFiles/cleanedMozartFiles/test/*.tsv"

# Beethoven
# TRAIN_TSV = "output/cleanedFiles/cleanedBeethovenFiles/train/*.tsv"
# TEST_TSV = "output/cleanedFiles/cleanedBeethovenFiles/test/*.tsv"

# Combined
# TRAIN_TSV = "output/cleanedFiles/combinedTrain/*.tsv"
# TEST_TSV = "output/cleanedFiles/combinedTest/*.tsv"


def _get_token(token_index: int, row_count: int, chords: List[str]):
    if token_index < 0:
        return "[START]"
    elif token_index >= row_count:
        return "[END]"
    else:
        return chords[token_index]  # previous or next chord


def extract_features(chords: List[str], chord: str, index: int, row_count: int
                     ) -> FeatureVector:
    """Extract feature vector for a single chord."""
    features: Dict[str, str] = {}

    features["chord"] = chord
    features["t-1"] = _get_token(index - 1, row_count, chords)
    features["t-2"] = _get_token(index - 2, row_count, chords)
    features["t-3"] = _get_token(index - 3, row_count, chords)
    # features["t-4"] = _get_token(index - 4, row_count, chords)
    features["t+1"] = _get_token(index + 1, row_count, chords)
    features["t+2"] = _get_token(index + 2, row_count, chords)
    features["t+3"] = _get_token(index + 3, row_count, chords)
    # features["t+4"] = _get_token(index + 4, row_count, chords)

    # Bigrams
    features["t-2^t-1"] = f"{_get_token(index - 2, row_count, chords)}^{_get_token(index - 1, row_count, chords)}"
    features["t+1^t+2"] = f"{_get_token(index + 1, row_count, chords)}^{_get_token(index + 2, row_count, chords)}"
    features["t-1^t+1"] = f"{_get_token(index - 1, row_count, chords)}^{_get_token(index + 1, row_count, chords)}"

    # Trigrams
    features["t-3^t-2^t-1"] = f"{_get_token(index - 3, row_count, chords)}^{_get_token(index - 2, row_count, chords)}^{_get_token(index - 1, row_count, chords)}"
    features["t+1^t+2^t+3"] = f"{_get_token(index + 1, row_count, chords)}^{_get_token(index + 2, row_count, chords)}^{_get_token(index + 3, row_count, chords)}"
    features["t-2^t-1^t+1"] = f"{_get_token(index - 2, row_count, chords)}^{_get_token(index - 1, row_count, chords)}^{_get_token(index + 1, row_count, chords)}"
    features["t-1^t+1^t+2"] = f"{_get_token(index - 1, row_count, chords)}^{_get_token(index + 1, row_count, chords)}^{_get_token(index + 2, row_count, chords)}"

    # 4-grams tested but not ultimately used
    # features["t-4^t-3^t-2^t-1"] = f"{_get_token(index - 4, row_count, chords)}^{_get_token(index - 3, row_count, chords)}^{_get_token(index - 2, row_count, chords)}^{_get_token(index - 1, row_count, chords)}"
    # features["t+1^t+2^t+3^t+4"] = f"{_get_token(index + 1, row_count, chords)}^{_get_token(index + 2, row_count, chords)}^{_get_token(index + 3, row_count, chords)}^{_get_token(index + 4, row_count, chords)}"
    # features["t-2^t-1^t+1^t+2"] = f"{_get_token(index - 2, row_count, chords)}^{_get_token(index - 1, row_count, chords)}^{_get_token(index + 1, row_count, chords)}^{_get_token(index + 2, row_count, chords)}"

    # Positional features
    features["relative_position"] = index / row_count  # Normalized position
    features["absolute_position"] = index  # Absolute position

    # Features based on chord content
    features["contains_i_or_I"] = 1 if chord == 'i' or chord == "I" or \
        chord == "i6" or chord == "I6" else 0
    features["contains_V"] = 1 if chord == 'V' or chord == "V7" else 0
    features["contains_iv_or_IV"] = 1 if chord == 'iv' or chord == "IV" or \
        chord == "iv6" or chord == "IV6" else 0
    # German augmented 6th chord
    features["contains_Gr"] = 1 if 'Gr' in chord else 0
    # Italian augmented 6th chord
    features["contains_It"] = 1 if 'It' in chord else 0
    # Applied chords tested but not ultimately used
    # features["contains_App"] = 1 if '/' in chord else 0

    return features


def extract_features_file(path: str) -> Tuple[FeatureVectors, List[str]]:
    """Extracts feature vectors for an entire TSV file."""
    features: FeatureVectors = []
    labels: List[str] = []
    chords: List[str] = []

    # Read the chords first
    with open(path, "r") as source:
        reader = csv.DictReader(source, delimiter="\t")
        for row in reader:
            chords.append(row["**chords"])

    row_count = len(chords)

    # Count the unique function labels
    unique_labels = set()
    with open(path, "r") as source:
        for row in csv.DictReader(source, delimiter="\t"):
            unique_labels.add(row["**function"])

    # Skip files with only one function label
    if len(unique_labels) <= 1:
        print(
            f"Skipping file '{path}' because it has only one function label.")
        return features, labels

    # Count the rows
    with open(path, "r") as source:
        row_count = sum(1 for row in csv.DictReader(source, delimiter="\t"))

    with open(path, "r") as source:
        for index, row in enumerate(csv.DictReader(source, delimiter="\t")):
            labels.append(row["**function"])
            features.append(
                extract_features(
                    chords,
                    row["**chords"],
                    index,
                    row_count
                )
            )

    return features, labels


def main() -> None:
    correct: List[int] = []
    size: List[int] = []
    k = 0
    l = 0

    # Variables for the confusion matrices
    all_test_labels = []
    all_predictions = []

    for train_path in glob.iglob(TRAIN_TSV):
        # Extract training features and labels using
        # `extract_features_files`
        train_features, train_labels = extract_features_file(train_path)

        if not train_features:  # Check if train_features is empty
            print(f"Skipping file '{train_path}' because it has no samples.")
            print("count of skipped TRAINING files:", k)
            k += 1
            continue

        # Create a DictVectorizer object
        vectorizer = feature_extraction.DictVectorizer()

        # One-hot-encode the features using the object's `fit_transform`
        X_train = vectorizer.fit_transform(train_features)

        # Classifier 1
        # Create a LogisticRegression object
        classifier = linear_model.LogisticRegression(penalty="l1", C=10,
                                                     solver="liblinear",
                                                     max_iter=1000)

        # Fit the model using the object's `fit`, the vectorized
        # features, and the labels
        classifier.fit(X_train, train_labels)

        # Classifier 2
        """classifier = RandomForestClassifier(
            class_weight={'T': 5.5, 'P': 1, 'D': 3.5, '': 0, '**function': 0})
        classifier.fit(X_train, train_labels)"""

        # Classifier 3
        # Grid search for hyperparameter tuning
        """param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, None],
            'class_weight': [{'T': 5.5, 'P': 1, 'D': 3.5, '': 0, '**function': 0}]
        }
        grid_search = GridSearchCV(RandomForestClassifier(), param_grid,
                                   cv=2, scoring='accuracy')
        grid_search.fit(X_train, train_labels)
        best_classifier = grid_search.best_estimator_"""

    # Iterate over testing files
    for test_path in glob.iglob(TEST_TSV):
        # Extract test features and labels using `extract_features_file`
        test_features, test_labels = extract_features_file(test_path)

        if not test_features:  # Check if train_features is empty
            print(f"Skipping file '{test_path}' because it has no samples.")
            print("count of skipped TESTING files:", l)
            l += 1
            continue

        # One-hot-encode the features using the DictVectorizer's
        # `transform`
        X_test = vectorizer.transform(test_features)

        # Compute the number of correct predictions and append it to
        # `correct`
        # Predictions for classifiers 1 and 2
        predictions = classifier.predict(X_test)

        # Predictions for classifier 3
        # predictions = best_classifier.predict(X_test)

        correct_predictions = accuracy_score(test_labels, predictions,
                                             normalize=False)
        correct.append(correct_predictions)

        # Append the size of the test set to `size`
        size.append(len(test_labels))

        # Accumulate all predictions and labels for confusion matrices
        all_test_labels.extend(test_labels)
        all_predictions.extend(predictions)

    micro_avg_accuracy = sum(correct) / sum(size)
    homograph_accuracies = [c/s for c, s in zip(correct, size)]
    macro_avg_accuracy = np.mean(homograph_accuracies)

    # f-string rounding used
    print(f"Micro-averaged accuracy:\t{micro_avg_accuracy:.4f}")
    print(f"Macro-averaged accuracy:\t{macro_avg_accuracy:.4f}")

    # Confusion matrix
    # Define the class labels
    labels = ['T', 'P', 'D']

    # Compute the confusion matrix
    conf_matrix = multilabel_confusion_matrix(all_test_labels,
                                              all_predictions,
                                              labels=labels)

    # Print the confusion matrix
    for i, label in enumerate(labels):
        print(f"Confusion matrix for label '{label}':")
        print(conf_matrix[i])
        print()

    print("count of skipped TRAINING files:", k)
    print("count of skipped TESTING files:", l)


if __name__ == "__main__":
    main()
