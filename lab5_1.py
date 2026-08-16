import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


# A1 - normal kNN

cols = [
    "Class", "Alcohol", "Malicacid", "Ash", "Alcalinity_of_ash",
    "Magnesium", "Total_phenols", "Flavanoids",
    "Nonflavanoid_phenols", "Proanthocyanins",
    "Color_intensity", "Hue", "OD280_OD315", "Proline"
]


def distance(a, b):
    dist = 0
    for i in range(len(a)):
        dist += (a[i] - b[i]) ** 2
    return dist ** 0.5


def bubble_sort(arr, n):
    n = len(arr)
    for i in range(n):
        for j in range(n - i - 1):
            if arr[j][0] > arr[j + 1][0]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr


def selection_sort(arr, n):
    n = len(arr)
    for i in range(n):
        min_index = i
        for j in range(i + 1, n):
            if arr[j][0] < arr[min_index][0]:
                min_index = j
        arr[i], arr[min_index] = arr[min_index], arr[i]
    return arr


def insertion_sort(arr, n):
    n = len(arr)
    for i in range(1, n):
        current = arr[i]
        j = i - 1

        while j >= 0 and arr[j][0] > current[0]:
            arr[j + 1] = arr[j]
            j -= 1

        arr[j + 1] = current

    return arr


def sort_neighbors(arr, algorithm):
    if algorithm == "bubble":
        return bubble_sort(arr, len(arr))
    elif algorithm == "selection":
        return selection_sort(arr, len(arr))
    elif algorithm == "insertion":
        return insertion_sort(arr, len(arr))
    else:
        return bubble_sort(arr, len(arr))


def find_neighbors(train_data, test_point):
    neighbors = []

    for row in train_data:
        label = row[0]
        train_point = row[1:]

        dist = distance(train_point, test_point)
        neighbors.append([dist, label])

    return neighbors


def k_nearest_neighbors(neighbors, k, algorithm):
    neighbors = sort_neighbors(neighbors, algorithm)
    return neighbors[:k]


def predict_class_for_test_point(neighbors):
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
            tied_labels.append(labels[i])

    if len(tied_labels) == 1:
        return tied_labels[0]

    #to break the tie
    for neighbor in neighbors:
        for label in tied_labels:
            if neighbor[1] == label:
                return label


# A2 - weighted kNN

def weighted_vote(k_neighbors):
    labels = []
    weights = []

    for neighbor in k_neighbors:
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
            tied_labels.append(labels[i])

    if len(tied_labels) == 1:
        return tied_labels[0]

    # Tie-breaking: closest tied neighbor
    for neighbor in k_neighbors:
        for label in tied_labels:
            if neighbor[1] == label:
                return label


def predict_weighted(training_features, training_labels, test_features, k, algorithm):
    predictions = []

    training_data = []

    for i in range(len(training_features)):
        training_data.append(
            [training_labels[i]] + training_features[i]
        )

    for test_point in test_features:
        neighbors = find_neighbors(training_data, test_point)

        k_neighbors = k_nearest_neighbors(
            neighbors,
            k,
            algorithm
        )

        prediction = weighted_vote(k_neighbors)
        predictions.append(prediction)

    return predictions


def calculate_accuracy(predictions, actual_labels):
    correct_predictions = 0

    for i in range(len(actual_labels)):
        if predictions[i] == actual_labels[i]:
            correct_predictions += 1

    return correct_predictions / len(actual_labels)


# Common Dataset Functions

def load_wine_data():
    data_frame = pd.read_csv("wine.csv")
    return data_frame.values.tolist()


def prepare_two_class_data(data):
    selected_data = []

    for row in data:
        if row[0] == 1 or row[0] == 2:
            selected_data.append(row)

    X = []
    y = []

    for row in selected_data:
        y.append(row[0])
        X.append(row[1:])

    return X, y


# A3 - Train/Test Split

def create_train_test_split(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42
    )

    return X_train, X_test, y_train, y_test


# A4 - Train kNN

def train_package_knn(X_train, y_train, k):
    knn_classifier = KNeighborsClassifier(
        n_neighbors=k
    )

    knn_classifier.fit(
        X_train,
        y_train
    )

    return knn_classifier


# A5 - Test Accuracy

def get_package_accuracy(model, X_test, y_test):
    return model.score(
        X_test,
        y_test
    )


# A6 - Predict Test Vectors

def get_package_predictions(model, X_test):
    return model.predict(X_test)


# A7 - Custom fit(), predict(), score()

def fit(training_features, training_labels, k):
    model = {
        "training_features": training_features,
        "training_labels": training_labels,
        "k": k
    }

    return model


def predict(model, test_features):
    predictions = []

    for test_point in test_features:
        training_data = []

        for i in range(len(model["training_features"])):
            training_data.append(
                [model["training_labels"][i]]
                + model["training_features"][i]
            )

        neighbors = find_neighbors(
            training_data,
            test_point
        )

        k_neighbors = k_nearest_neighbors(
            neighbors,
            model["k"],
            "bubble"
        )

        prediction = predict_class_for_test_point(
            k_neighbors
        )

        predictions.append(prediction)

    return predictions


def score(model, test_features, actual_labels):
    predictions = predict(
        model,
        test_features
    )

    return calculate_accuracy(
        predictions,
        actual_labels
    )


# A8 - Custom kNN vs Package kNN


def predict_normal(training_features, training_labels, test_features, k, algorithm):
    predictions = []

    training_data = []

    for i in range(len(training_features)):
        training_data.append(
            [training_labels[i]] + training_features[i]
        )

    for test_point in test_features:
        neighbors = find_neighbors(
            training_data,
            test_point
        )

        k_neighbors = k_nearest_neighbors(
            neighbors,
            k,
            algorithm
        )

        prediction = predict_class_for_test_point(
            k_neighbors
        )

        predictions.append(prediction)

    return predictions


def compare_custom_and_package_knn(
    X_train,
    y_train,
    X_test,
    y_test,
    k_values,
    algorithm
):
    custom_accuracies = []
    package_accuracies = []

    for k in k_values:
        custom_predictions = predict_normal(
            X_train,
            y_train,
            X_test,
            k,
            algorithm
        )

        custom_accuracy = calculate_accuracy(
            custom_predictions,
            y_test
        )

        custom_accuracies.append(custom_accuracy)

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

        package_accuracies.append(package_accuracy)

    return custom_accuracies, package_accuracies


# A9 - Normal kNN vs Weighted kNN


def compare_normal_and_weighted_knn(
    X_train,
    y_train,
    X_test,
    y_test,
    k_values,
    algorithm
):
    normal_accuracies = []
    weighted_accuracies = []

    for k in k_values:
        normal_predictions = predict_normal(
            X_train,
            y_train,
            X_test,
            k,
            algorithm
        )

        normal_accuracy = calculate_accuracy(
            normal_predictions,
            y_test
        )

        normal_accuracies.append(normal_accuracy)

        weighted_predictions = predict_weighted(
            X_train,
            y_train,
            X_test,
            k,
            algorithm
        )

        weighted_accuracy = calculate_accuracy(
            weighted_predictions,
            y_test
        )

        weighted_accuracies.append(weighted_accuracy)

    return normal_accuracies, weighted_accuracies



# MAIN PROGRAM


def main():


    # A1 - Load Dataset


    df = pd.read_csv(
        "wine.data",
        header=None,
        names=cols
    )

    df.to_csv(
        "wine.csv",
        index=False
    )

    data = df.values.tolist()

    train_data = data[:150]
    test_data = data[150:]

    k = 5

    sorting_algorithm = "bubble"

  


    print("A1 - CUSTOM kNN")


    print(
        "Sorting Algorithm =",
        sorting_algorithm
    )

    print(
        "k =",
        k
    )

    print()

    for i in range(len(test_data)):
        test_row = test_data[i]

        actual_class = test_row[0]
        test_point = test_row[1:]

        neighbors = find_neighbors(
            train_data,
            test_point
        )

        k_neighbors = k_nearest_neighbors(
            neighbors,
            k,
            sorting_algorithm
        )

        prediction = predict_class_for_test_point(
            k_neighbors
        )

        print(
            "Test Point", i + 1,
            "Actual Class =", actual_class,
            "Predicted Class =", prediction
        )


    # Prepare Class 1 and Class 2 Data for A2-A9

    X, y = prepare_two_class_data(data)

    X_train, X_test, y_train, y_test = (
        create_train_test_split(X, y)
    )


    # A2 - Weighted kNN

    print()

    print("A2 - WEIGHTED kNN")
   

    example_neighbors = [
        [0.5, 1],
        [1.0, 2],
        [2.0, 2]
    ]

    weighted_prediction = weighted_vote(
        example_neighbors
    )

    print(
        "Weighted kNN Prediction =",
        weighted_prediction
    )

    weighted_predictions = predict_weighted(
        X_train,
        y_train,
        X_test,
        3,
        sorting_algorithm
    )

    weighted_accuracy = calculate_accuracy(
        weighted_predictions,
        y_test
    )

    print(
        "Weighted kNN Accuracy =",
        weighted_accuracy
    )

    print(
        "Weighted kNN Accuracy Percentage =",
        weighted_accuracy * 100,
        "%"
    )



    # A3 - Train/Test Split

    print()
  
    print("A3 - TRAIN / TEST SPLIT")


    print(
        "Total Samples =",
        len(X)
    )

    print(
        "Training Samples =",
        len(X_train)
    )

    print(
        "Testing Samples =",
        len(X_test)
    )

    print()
    print("Training Labels =")
    print(y_train)

    print()
    print("Testing Labels =")
    print(y_test)



    # A4 - Train kNN with k = 3
 

    print()
   
    print("A4 - TRAIN kNN WITH k = 3")


    package_model = train_package_knn(
        X_train,
        y_train,
        3
    )

    print(
        "kNN classifier trained successfully."
    )

    print(
        "Value of k =",
        3
    )

    print(
        "Training samples =",
        len(X_train)
    )

    print(
        "Testing samples =",
        len(X_test)
    )


    # A5 - Test Accuracy
 

    print()
  
    print("A5 - TEST ACCURACY")


    package_accuracy = get_package_accuracy(
        package_model,
        X_test,
        y_test
    )

    print(
        "k =",
        3
    )

    print(
        "Accuracy =",
        package_accuracy
    )

    print(
        "Accuracy Percentage =",
        package_accuracy * 100,
        "%"
    )


    # A6 - Predict Test Vectors

    print()
    print("A6 - PREDICT TEST VECTORS")

    predictions = get_package_predictions(
        package_model,
        X_test
    )

    for i in range(len(X_test)):
        print(
            "Test Point =", i + 1,
            "Actual Class =", y_test[i],
            "Predicted Class =", predictions[i]
        )


    # A7 - fit(), predict(), score()

    print()
    print("A7 - fit(), predict(), score()")

    a7_training_features = [
        [1.0, 2.0],
        [1.2, 1.8],
        [5.0, 6.0],
        [5.2, 5.8]
    ]

    a7_training_labels = [
        1,
        1,
        2,
        2
    ]

    a7_test_features = [
        [1.1, 2.1],
        [5.1, 6.1]
    ]

    a7_test_labels = [
        1,
        2
    ]

    model = fit(
        a7_training_features,
        a7_training_labels,
        3
    )

    a7_predictions = predict(
        model,
        a7_test_features
    )

    a7_accuracy = score(
        model,
        a7_test_features,
        a7_test_labels
    )

    print(
        "Predictions =",
        a7_predictions
    )

    print(
        "Actual Labels =",
        a7_test_labels
    )

    print(
        "Accuracy =",
        a7_accuracy
    )


    # A8 - Custom vs Package


    print()
    print("A8 - CUSTOM kNN vs PACKAGE kNN")

    k_values = []

    for k in range(1, 21):
        k_values.append(k)

    custom_accuracies, package_accuracies = (
        compare_custom_and_package_knn(
            X_train,
            y_train,
            X_test,
            y_test,
            k_values,
            sorting_algorithm
        )
    )

    for i in range(len(k_values)):
        print(
            "k =", k_values[i],
            "Custom =", custom_accuracies[i],
            "Package =", package_accuracies[i]
        )

    plt.figure(figsize=(10, 6))

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

    plt.xlabel("Value of k")
    plt.ylabel("Accuracy")
    plt.title("Custom kNN vs Scikit-learn kNN")
    plt.legend()
    plt.grid(True)
    plt.show()


    # A9 - Normal vs Weighted

    print()
    print("A9 - WEIGHTED vs NORMAL kNN")

    normal_accuracies, weighted_accuracies = (
        compare_normal_and_weighted_knn(
            X_train,
            y_train,
            X_test,
            y_test,
            k_values,
            sorting_algorithm
        )
    )

    for i in range(len(k_values)):
        print(
            "k =", k_values[i],
            "Normal kNN =", normal_accuracies[i],
            "Weighted kNN =", weighted_accuracies[i]
        )

    plt.figure(figsize=(10, 6))

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

    plt.xlabel("Value of k")
    plt.ylabel("Accuracy")
    plt.title("Normal kNN vs Weighted kNN")
    plt.legend()
    plt.grid(True)
    plt.show()


# PROGRAM ENTRY POINT

if __name__ == "__main__":
    main()
