# ============================================================
# A3 - TRAIN / TEST SPLIT
# ============================================================

import pandas as pd

from sklearn.model_selection import train_test_split

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



# Main Program
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

print(
    "Training Labels ="
)

print(y_train)

print()

print(
    "Testing Labels ="
)

print(y_test)