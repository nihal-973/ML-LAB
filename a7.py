# A7 - IMPLEMENT fit(), predict(), score()
# Distance Function
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
    training_features,
    training_labels,
    test_point
):
    neighbors = []
    for i in range(
        len(training_features)
    ):
        dist = distance(
            training_features[i],
            test_point
        )
        neighbors.append(
            [
                dist,
                training_labels[i]
            ]
        )
    return neighbors
# predict One Class
def predict_class(neighbors):
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
        if found == False:
            labels.append(label)
            counts.append(1)
    # to find highest vote
    highest_count = counts[0]
    for count in counts:
        if count > highest_count:
            highest_count = count
    #tie
    tied_labels = []
    for i in range(len(labels)):
        if counts[i] == highest_count:
            tied_labels.append(
                labels[i]
            )
    # No tie
    if len(tied_labels) == 1:
        return tied_labels[0]
    # Tie-breaking:
    # choose closest neighbor
    for neighbor in neighbors:
        for label in tied_labels:
            if neighbor[1] == label:
                return label
# fit()
def fit(
    training_features,
    training_labels,
    k
):
    model = [
        training_features,
        training_labels,
        k
    ]
    return model
# predict()
def predict(
    model,
    test_features
):
    training_features = model[0]
    training_labels = model[1]
    k = model[2]
    predictions = []
    for test_point in test_features:
        neighbors = find_neighbors(
            training_features,
            training_labels,
            test_point
        )
        neighbors = bubble_sort(
            neighbors
        )
        k_neighbors = neighbors[:k]
        prediction = predict_class(
            k_neighbors
        )
        predictions.append(
            prediction
        )
    return predictions
# score()
def score(
    model,
    test_features,
    actual_labels
):
    predictions = predict(
        model,
        test_features
    )
    correct_predictions = 0
    for i in range(
        len(actual_labels)
    ):
        if predictions[i] == actual_labels[i]:
            correct_predictions += 1
    accuracy = (
        correct_predictions/len(actual_labels)
    )
    return accuracy
# MAIN PROGRAM
# Training Data
training_features = [
    [1.0, 2.0],[1.2, 1.8],[5.0, 6.0],[5.2, 5.8]
]
# Training Labels
training_labels = [1,1,2,2]
# Test Data
test_features = [[1.1, 2.1],[5.1, 6.1]]
# Test Labels
test_labels = [1,2]
# Train Model
model = fit(
    training_features,
    training_labels,
    k=3
)
# Predict
predictions = predict(
    model,
    test_features
)
# Calculate Accuracy
accuracy = score(
    model,
    test_features,
    test_labels
)
# Display Results
print("A7")
print(
    "Predictions =",
    predictions
)
print(
    "Actual Labels =",
    test_labels
)
print(
    "Accuracy =",
    accuracy
)