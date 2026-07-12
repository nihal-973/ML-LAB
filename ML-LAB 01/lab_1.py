import random
from collections import Counter

def count_vowels_consonants(input_string):
    vowels = 'aeiouAEIOU'
    vowel_count = 0
    consonant_count = 0
    
    for char in input_string:
        if char.isalpha():
            if char in vowels:
                vowel_count += 1
            else:
                consonant_count += 1
    
    return vowel_count, consonant_count

def multiply_matrices(A, B):
    if not A or not B:
        return "Error: Matrices cannot be empty"
    
    rows_A = len(A)
    cols_A = len(A[0])
    rows_B = len(B)
    cols_B = len(B[0])
    
    for row in A:
        if len(row) != cols_A:
            return "Error: Matrix A has inconsistent row lengths"
    for row in B:
        if len(row) != cols_B:
            return "Error: Matrix B has inconsistent row lengths"
    
    if cols_A != rows_B:
        return f"Error: Matrices cannot be multiplied. A has {cols_A} columns, B has {rows_B} rows."
    
    result = [[0 for _ in range(cols_B)] for _ in range(rows_A)]
    
    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                result[i][j] += A[i][k] * B[k][j]
    
    return result

def find_common_elements(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    common = set1.intersection(set2)
    return len(common), list(common)

def transpose_matrix(matrix):
    if not matrix:
        return "Error: Matrix cannot be empty"
    
    rows = len(matrix)
    cols = len(matrix[0])
    
    for row in matrix:
        if len(row) != cols:
            return "Error: Matrix has inconsistent row lengths"
    
    transposed = [[matrix[i][j] for i in range(rows)] for j in range(cols)]
    return transposed

def generate_random_numbers(count=100, low=100, high=150):
    return [random.randint(low, high) for _ in range(count)]

def calculate_statistics(numbers):
    if not numbers:
        return 0, 0, 0
    
    n = len(numbers)
    sorted_nums = sorted(numbers)
    
    mean = sum(numbers) / n
    
    if n % 2 == 0:
        median = (sorted_nums[n//2 - 1] + sorted_nums[n//2]) / 2
    else:
        median = sorted_nums[n//2]
    
    counter = Counter(numbers)
    max_count = max(counter.values())
    mode = [num for num, count in counter.items() if count == max_count]
    mode = min(mode) if mode else 0
    
    return mean, median, mode

def main():
    print("=" * 50)
    print("ASSIGNMENT OUTPUT")
    print("=" * 50)
    
    print("Q1 : NO of vowels")
    test_string = "TestString"
    print(f"Input: {test_string}")
    v, c = count_vowels_consonants(test_string)
    print(f"Vowels: {v}")
    print(f"Consonants: {c}")
    
    print("Q2 : Multiply matrix")
    A = [[1, 2, 3], [4, 5, 6]]
    B = [[7, 8], [9, 10], [11, 12]]
    print("Matrix A:")
    for row in A:
        print(row)
    print("Matrix B:")
    for row in B:
        print(row)
    
    result = multiply_matrices(A, B)
    if type(result) == str:
        print(result)
    else:
        print("Product AB:")
        for row in result:
            print(row)
    
    print("Q3 : Common Elements")
    list1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    list2 = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    print(f"List 1: {list1}")
    print(f"List 2: {list2}")
    count, common = find_common_elements(list1, list2)
    print(f"Common count: {count}")
    print(f"Common elements: {common}")
    
    print("Q4: Matrix Transpose")
    matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    print("Original:")
    for row in matrix:
        print(row)
    transposed = transpose_matrix(matrix)
    if type(transposed) == str:
        print(transposed)
    else:
        print("Transposed:")
        for row in transposed:
            print(row)
    
    print("Q5 : Mean, Mode, Median")
    nums = generate_random_numbers()
    print(f"Generated 100 numbers between 100-150")
    print(f"First 10: {nums[:10]}")
    mean, median, mode = calculate_statistics(nums)
    print(f"Mean: {mean:.2f}")
    print(f"Median: {median:.2f}")
    print(f"Mode: {mode}")
    
    print("\n" + "=" * 50)
if __name__ == "__main__":
    main()