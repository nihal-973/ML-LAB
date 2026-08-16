# A8 - CUSTOM kNN vs PACKAGE kNN
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
# Distance
def distance(a, b):
    dist = 0
    for i in range(len(a)):
        dist += (
            (a[i] - b[i]) ** 2
        )
    return dist ** 0.5
# Bubble Sort
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
# Find Neighbors
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
# K Nearest Neighbors
def k_nearest_neighbors(
    neighbors,
    k
):
    neighbors = bubble_sort(
        neighbors
    )
    return neighbors[:k]
# Predict Class
def predict_class(
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
    for neighbor in neighbors:
        for label in tied_labels:
            if neighbor[1] == label:
                return label
# Custom fit
def fit(
    X_train,
    y_train,
    k
):
    model = {
        "X_train": X_train,
        "y_train": y_train,
        "k": k
    }
    return model
# Custom predict
def predict(
    model,
    X_test
):
    predictions = []
    training_data = []
    for i in range(
        len(model["X_train"])
    ):
        training_data.append(
            [
                model["y_train"][i]
            ]
            +
            model["X_train"][i]
        )
    for test_point in X_test:
        neighbors = find_neighbors(
            training_data,
            test_point
        )
        k_neighbors = k_nearest_neighbors(
            neighbors,
            model["k"]
        )
        prediction = predict_class(
            k_neighbors
        )
        predictions.append(
            prediction
        )
    return predictions
# Custom score
def score(
    model,
    X_test,
    y_test
):
    predictions = predict(
        model,
        X_test
    )
    correct = 0
    for i in range(
        len(y_test)
    ):
        if predictions[i] == y_test[i]:
            correct += 1
    return (
        correct /
        len(y_test)
    )
# Load Dataset
data_frame = pd.read_csv(
    "wine.csv"
)
data = data_frame.values.tolist()
# cls 1 or 2 selection
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
custom_accuracies = []
package_accuracies = []
# Experiment
for k in k_values:
    custom_model = fit(
        X_train,
        y_train,
        k
    )
    custom_accuracy = score(
        custom_model,
        X_test,
        y_test
    )
    custom_accuracies.append(
        custom_accuracy
    )
    # Package kNN
    package_model = KNeighborsClassifier(
        n_neighbors=k
    )
    package_model.fit(
        X_train,
        y_train
    )
    package_accuracy = package_model.score(
        X_test,
        y_test
    )
    package_accuracies.append(
        package_accuracy
    )
print("A8 - CUSTOM kNN vs PACKAGE kNN")
for i in range(
    len(k_values)
):
    print(
        "k =", k_values[i],
        "Custom =", custom_accuracies[i],
        "Package =", package_accuracies[i]
    )
plt.figure(
    figsize=(10, 6)
)
plt.plot(
    k_values,
    custom_accuracies,
    marker="o",
    label="Custom kNN"
)
plt.plot(
    k_values,
    package_accuracies,
    marker="s",
    label="Scikit-learn kNN"
)
plt.xlabel(
    "Value of k"
)
plt.ylabel(
    "Accuracy"
)
plt.title(
    "Custom kNN vs Scikit-learn kNN"
)
plt.legend()
plt.grid(True)
plt.show()