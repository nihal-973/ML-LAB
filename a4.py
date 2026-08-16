# A4 - TRAIN kNN WITH k = 3
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
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

# Create kNN Classifier

neigh = KNeighborsClassifier(
    n_neighbors=3
)

# Train Classifier

neigh.fit(
    X_train,
    y_train
)

# Main Program
print("A4 -WITH k = 3")

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