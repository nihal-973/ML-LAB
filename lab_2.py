import pandas as pd 
import numpy as np

df = pd.read_excel("Lab Session Data.xlsx",sheet_name = "Purchase data")

print("purchase data:\n")
print(df.head())

X = df[["Candies (#)","Mangoes (Kg)","Milk Packets (#)"]]
Y = df[["Payment (Rs)"]]
x = X.values
y = Y.values

print("features matrix :")
print(x)
print("output matrix:")
print(y)

dimension = x.shape[1]
print(dimension)
num_vectors = x.shape[0]
print(num_vectors)


rank = np.linalg.matrix_rank(x)
print(rank)

X_pinv = np.linalg.pinv(X)
print("\nPseudo Inverse of X:")

print(X_pinv)
