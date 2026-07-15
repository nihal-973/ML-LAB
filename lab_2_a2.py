import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_excel("Lab Session Data.xlsx", sheet_name="Purchase data")

df["Class"] = df["Payment (Rs)"].apply(lambda x: "RICH" if x > 200 else "POOR")

X = df[["Candies (#)", "Mangoes (Kg)", "Milk Packets (#)"]]
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

df["Predicted Class"] = model.predict(X)

print("\nFinal Results:")
print(df[["Customer", "Payment (Rs)", "Class", "Predicted Class"]])