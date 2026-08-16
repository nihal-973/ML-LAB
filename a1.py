# a1 
import pandas as pd
cols = [
    "Class",
    "Alcohol",
    "Malicacid",
    "Ash",
    "Alcalinity_of_ash",
    "Magnesium",
    "Total_phenols",
    "Flavanoids",
    "Nonflavanoid_phenols",
    "Proanthocyanins",
    "Color_intensity",
    "Hue",
    "OD280_OD315",
    "Proline"
]

# Distance Function

def distance(a, b):

    dist = 0

    for i in range(len(a)):
        dist += (a[i] - b[i]) ** 2

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

# Selection Sort


def selection_sort(arr):

    n = len(arr)

    for i in range(n):

        min_index = i

        for j in range(i + 1, n):

            if arr[j][0] < arr[min_index][0]:

                min_index = j

        arr[i], arr[min_index] = (
            arr[min_index],
            arr[i]
        )

    return arr

# Insertion Sort


def insertion_sort(arr):

    n = len(arr)

    for i in range(1, n):

        current = arr[i]

        j = i - 1

        while j >= 0 and arr[j][0] > current[0]:

            arr[j + 1] = arr[j]

            j -= 1

        arr[j + 1] = current

    return arr

# Select Sorting Algorithm


def sort_neighbors(arr, algorithm):

    if algorithm == "bubble":

        return bubble_sort(arr)

    elif algorithm == "selection":

        return selection_sort(arr)

    elif algorithm == "insertion":

        return insertion_sort(arr)

    else:

        return bubble_sort(arr)


# finfing Neighbors

def find_neighbors(train_data, test_point):

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



# K nearest neighbrs

def k_nearest_neighbors(
    neighbors,
    k,
    algorithm
):

    neighbors = sort_neighbors(
        neighbors,
        algorithm
    )

    return neighbors[:k]

# Predict Class


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


    # No tie
    if len(tied_labels) == 1:

        return tied_labels[0]

    # Choose class of closest tied neighbor


    for neighbor in neighbors:

        for label in tied_labels:

            if neighbor[1] == label:

                return label

# Load wine.data

df = pd.read_csv(
    "wine.data",
    header=None,
    names=cols
)


df.to_csv(
    "wine.csv",
    index=False
)
# Convert DataFrame to List
data = df.values.tolist()
# Split Dataset
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


    prediction = predict_class(
        k_neighbors
    )


    print(
        "Test Point =", i + 1,
        "Actual Class =", actual_class,
        "Predicted Class =", prediction
    )