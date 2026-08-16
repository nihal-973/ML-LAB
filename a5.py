# A5 - TEST ACCURACY


import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
# Load Dataset

data_frame = pd.read_csv(
    "wine.csv"
)

data = data_frame.values.tolist()

selected_data = []

for row in data:

    if row[0] == 1 or row[0] == 2:

        selected_data.append(row)
# Separate Features and Labels

X = []
y = []

for row in selected_data:

    y.append(row[0])

    X.append(row[1:])

# Train-Test Split
X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42
    )
)
# Create and Train kNN

neigh = KNeighborsClassifier(
    n_neighbors=3
)

neigh.fit(
    X_train,
    y_train
)
# Calculate Accuracy

accuracy = neigh.score(
    X_test,
    y_test
)
# Main Program
print("A5 - TEST ACCURACY")

print(
    "k =",
    3
)

print(
    "Accuracy =",
    accuracy
)

print(
    "Accuracy Percentage =",
    accuracy * 100,
    "%"
)