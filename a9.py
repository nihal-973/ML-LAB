# a9
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
def distance(a, b):
    dist = 0
    for i in range(len(a)):
        dist += (
            (a[i] - b[i]) ** 2
        )
    return dist ** 0.5
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(n - i - 1):
            if arr[j][0] > arr[j + 1][0]:
                arr[j], arr[j + 1] = (
                    arr[j + 1],
                    arr[j]
                )
    return arr
def find_neighbors(
    train_data,
    test_point
):
    neighbors = []
    for row in train_data:
        label = row[0]
        train_point = row[1:]
        dist = distance(
            train_point,
            test_point
        )
        neighbors.append(
            [dist, label]
        )
    return neighbors
def k_nearest_neighbors(
    neighbors,
    k
):
    neighbors = bubble_sort(
        neighbors
    )

    return neighbors[:k]


def normal_vote(
    neighbors
):
    labels = []
    counts = []
    for neighbor in neighbors:
        label = neighbor[1]
        found = False
        for i in range(len(labels)):
            if labels[i] == label:
                counts[i] += 1
                found = True
                break
        if not found:
            labels.append(label)
            counts.append(1)
    highest_count = counts[0]
    for count in counts:
        if count > highest_count:
            highest_count = count
    tied_labels = []
    for i in range(len(labels)):
        if counts[i] == highest_count:
            tied_labels.append(
                labels[i]
            )
    if len(tied_labels) == 1:
        return tied_labels[0]
    # tie
    for neighbor in neighbors:
        for label in tied_labels:
            if neighbor[1] == label:
                return label
def weighted_vote(
    neighbors
):
    labels = []
    weights = []
    for neighbor in neighbors:
        distance_value = neighbor[0]
        label = neighbor[1]
        if distance_value == 0:
            weight = 1000000
        else:
            weight = 1 / distance_value
        found = False
        for i in range(len(labels)):
            if labels[i] == label:
                weights[i] += weight
                found = True
                break
        if not found:
            labels.append(label)
            weights.append(weight)
    highest_weight = weights[0]
    for weight in weights:
        if weight > highest_weight:
            highest_weight = weight
    tied_labels = []
    for i in range(len(labels)):
        if weights[i] == highest_weight:
            tied_labels.append(
                labels[i]
            )
    if len(tied_labels) == 1:
        return tied_labels[0]
    # tie
    for neighbor in neighbors:
        for label in tied_labels:
            if neighbor[1] == label:
                return label
def predict_normal(
    X_train,
    y_train,
    X_test,
    k
):
    predictions = []
    training_data = []
    for i in range(
        len(X_train)
    ):
        training_data.append(
            [
                y_train[i]
            ]
            +
            X_train[i]
        )
    for test_point in X_test:
        neighbors = find_neighbors(
            training_data,
            test_point
        )
        k_neighbors = k_nearest_neighbors(
            neighbors,
            k
        )
        prediction = normal_vote(
            k_neighbors
        )
        predictions.append(
            prediction
        )
    return predictions
def predict_weighted(
    X_train,
    y_train,
    X_test,
    k
):
    predictions = []
    training_data = []
    for i in range(
        len(X_train)
    ):
        training_data.append(
            [
                y_train[i]
            ]
            +
            X_train[i]
        )
    for test_point in X_test:
        neighbors = find_neighbors(
            training_data,
            test_point
        )
        k_neighbors = k_nearest_neighbors(
            neighbors,
            k
        )
        prediction = weighted_vote(
            k_neighbors
        )
        predictions.append(
            prediction
        )
    return predictions
# Accuracy
def calculate_accuracy(
    predictions,
    actual_labels
):
    correct = 0
    for i in range(
        len(actual_labels)
    ):
        if predictions[i] == actual_labels[i]:
            correct += 1
    return (
        correct /len(actual_labels)
    )
# Load Dataset
data_frame = pd.read_csv(
    "wine.csv"
)
data = data_frame.values.tolist()
# Select Class 1 and Class 2
selected_data = []
for row in data:
    if row[0] == 1 or row[0] == 2:
        selected_data.append(row)
# Features and Labels
X = []
y = []
for row in selected_data:
    y.append(row[0])
    X.append(row[1:])
# Train/Test Split
X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42
    )
)
# k Values
k_values = []
for k in range(1, 21):
    k_values.append(k)
normal_accuracies = []
weighted_accuracies = []
# Experiment
for k in k_values:
    # Normal kNN
    normal_predictions = predict_normal(
        X_train,
        y_train,
        X_test,
        k
    )
    normal_accuracy = calculate_accuracy(
        normal_predictions,
        y_test
    )
    normal_accuracies.append(
        normal_accuracy
    )
    # Weighted kNN
    weighted_predictions = predict_weighted(
        X_train,
        y_train,
        X_test,
        k
    )
    weighted_accuracy = calculate_accuracy(
        weighted_predictions,
        y_test
    )
    weighted_accuracies.append(
        weighted_accuracy
    )
print("A9 - WEIGHTED vs NORMAL kNN")
for i in range(
    len(k_values)
):
    print(
        "k =", k_values[i],
        "Normal kNN =", normal_accuracies[i],
        "Weighted kNN =", weighted_accuracies[i]
    )
plt.figure(
    figsize=(10, 6)
)
plt.plot(
    k_values,
    normal_accuracies,
    marker="o",
    label="Normal kNN"
)
plt.plot(
    k_values,
    weighted_accuracies,
    marker="s",
    label="Weighted kNN"
)
plt.xlabel(
    "Value of k"
)
plt.ylabel(
    "Accuracy"
)
plt.title(
    "Normal kNN vs Weighted kNN"
)
plt.legend()
plt.grid(True)
plt.show()