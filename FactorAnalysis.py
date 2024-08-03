import csv
import random
import math

# Read CSV file and store data in a list
csv_file_path = 'aps_failure_training_set.csv'

data_list = []
with open(csv_file_path, 'r') as file:
    csv_reader = csv.reader(file, delimiter=';')
    next(csv_reader)  # Skip header row
    for row in csv_reader:
        data_list.append(row[0].split(','))

# Convert strings to floats and handle 'na', 'neg', 'pos'
data_matrix = []
for row in data_list:
    row = [-1 if x == 'na' else (0 if x == 'neg' else (1 if x == 'pos' else float(x))) for x in row]
    data_matrix.append(row)

# Delete the 91st column
data_matrix = [row[:91] + row[92:] for row in data_matrix]

X = [row[2:] for row in data_matrix]  # Features
y = [int(row[1]) for row in data_matrix]  # Target

# Impute missing values with column means
def mean(arr):
    return sum(arr) / len(arr) if len(arr) > 0 else 0

for j in range(len(X[0])):
    col_values = [X[i][j] for i in range(len(X)) if X[i][j] != -1]
    col_mean = mean(col_values)
    for i in range(len(X)):
        if X[i][j] == -1:
            X[i][j] = col_mean

# Standardize features
def std_dev(arr):
    m = mean(arr)
    return (sum((x - m) ** 2 for x in arr) / len(arr)) ** 0.5 if len(arr) > 0 else 0

means = [mean([X[i][j] for i in range(len(X))]) for j in range(len(X[0]))]
stds = [std_dev([X[i][j] for i in range(len(X))]) for j in range(len(X[0]))]
X_scaled = [[(X[i][j] - means[j]) / stds[j] for j in range(len(X[0]))] for i in range(len(X))]

# Implement Factor Analysis manually

# Assuming 'n_components' is 110
n_components = 110

# Initialize factor analysis components randomly
components = [[random.uniform(-1, 1) for _ in range(len(X_scaled[0]))] for _ in range(n_components)]

# Update factor analysis components iteratively (using a simple method)
num_iterations = 100  # Define the number of iterations
learning_rate = 0.01  # Define the learning rate

for _ in range(num_iterations):
    for k in range(n_components):
        for j in range(len(X_scaled[0])):
            gradient = 0
            for i in range(len(X_scaled)):
                residual = X_scaled[i][j] - sum(components[k][l] * X_scaled[i][l] for l in range(len(X_scaled[0])) if l != j)
                gradient += residual * X_scaled[i][j]
            components[k][j] += learning_rate * gradient

# Perform dimensionality reduction manually
X_reduced = [[sum(X_scaled[i][j] * components[k][j] for j in range(len(X_scaled[0]))) for k in range(n_components)] for i in range(len(X_scaled))]


