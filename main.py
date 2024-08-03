import csv
import numpy as np
from LDA import LDA
from PCA import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score

training_csv_file_path = 'aps_failure_training_set.csv'
test_csv_file_path = 'aps_failure_test_set.csv'

with open(training_csv_file_path, 'r') as file:
    csv_reader = csv.reader(file, delimiter=';')
    next(csv_reader)  # skipping header row
    data_list_train = list(csv_reader)

with open(test_csv_file_path, 'r') as file:
    csv_reader = csv.reader(file, delimiter=';')
    next(csv_reader)  # skipping header row
    data_list_test = list(csv_reader)

data_matrix_train = np.zeros(shape=(60000, 172))
data_matrix_test = np.zeros(shape=(16000, 171))

# split function converts all strings to floats 
# na temporarily replaced with -1, so we know it's not a natural data and split can convert values to float
for i in range(len(data_list_train)):
    data_matrix_train[i] = data_list_train[i][0].replace('neg', '0').replace('na', '-1').replace('pos', '1').split(',')

for i in range(len(data_list_test)):
    data_matrix_test[i] = data_list_test[i][0].replace('neg', '0').replace('na', '-1').replace('pos', '1').split(',')

# deleted 91st column (90th in testing data) because it has a single value and all 'na's. With the data imputation
# technique used, it was creating a feature with zero standard deviation, so I deleted it
data_matrix_train = np.delete(data_matrix_train, 91, axis=1)
data_matrix_test = np.delete(data_matrix_test, 90, axis=1)

X = data_matrix_train[:, 2:]  # shape = (60000, 170)
y = data_matrix_train[:, 1].astype(int)  # shape = (60000,)

X_test = data_matrix_test[:, 1:]

# scan all rows for selected column, if -1 found na_count incremented by one
# for natural values add to col_mean to find the mean of the column then replace -1s with mean
for j in range(X.shape[1]):
    col_mean = 0.0
    na_count = 0
    for i in range(X.shape[0]):
        if X[i, j] == -1.0:
            na_count += 1
        else:
            col_mean += X[i, j]
    col_mean = col_mean / (X.shape[0] - na_count)
    for i in range(X.shape[0]):
        if X[i, j] == -1.0:
            X[i, j] = col_mean

for j in range(X_test.shape[1]):
    col_mean = 0.0
    na_count = 0
    for i in range(X_test.shape[0]):
        if X_test[i, j] == -1.0:
            na_count += 1
        else:
            col_mean += X_test[i, j]
    col_mean = col_mean / (X_test.shape[0] - na_count)
    for i in range(X_test.shape[0]):
        if X_test[i, j] == -1.0:
            X_test[i, j] = col_mean


def standardize_data(data):
    # calculating mean and standard deviation for each feature
    means = [sum(col) / len(col) for col in zip(*data)]
    stdevs = [((sum((x - mean) ** 2 for x in col)) / len(col)) ** 0.5 for mean, col in zip(means, zip(*data))]

    # standardizing the data
    standardized_data = [[(x - mean) / stdev for x, mean, stdev in zip(row, means, stdevs)] for row in data]

    return np.array(standardized_data)


X_stdman = standardize_data(X)
lda = LDA(100)
# fit_transform outputs complex values for some reason, so we use np.real to only take the real parts of the numbers
lda.fit(X_stdman, y)
X_projected_lda = np.real(lda.transform(X_stdman))
X_projected_lda = standardize_data(X_projected_lda)

X_std_test = standardize_data(X_test)
X_projected_test = np.real(lda.transform(X_std_test))
X_projected_test = standardize_data(X_projected_test)

X_train, X_valid, y_train, y_valid = train_test_split(standardize_data(X_projected_lda), y, test_size=0.2, shuffle=True)

model = LogisticRegression(random_state=1502, max_iter=1000000, class_weight={0: 1, 1: 5})
model.fit(X_train, y_train)
y_valid_pred = model.predict(X_valid)

conf_matrix = confusion_matrix(y_valid, y_valid_pred)
classification_rep = classification_report(y_valid, y_valid_pred)
auc_score = roc_auc_score(y_valid, y_valid_pred)

print("\nConfusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", classification_rep)
print("\n Area Under Curve: \n", auc_score)

y_test_pred = model.predict(standardize_data(X_projected_test))
y_test_pred = np.where(y_test_pred == 0, 'neg', 'pos')

with open('submission.csv', mode='w', newline='') as csvfile:
    fieldnames = ['id', 'class']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for i in range(y_test_pred.shape[0]):
        writer.writerow({"id": i+1, "class": y_test_pred[i]})
