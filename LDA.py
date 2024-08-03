import numpy as np


class LDA:
    def __init__(self, num_linear_discnants):
        self.num_linear_discnants = num_linear_discnants
        self.linear_discnants = None

    def fit(self, X, y):
        num_feats = X.shape[1]  # get number of features
        # initialize scatter within array and scatter between array with zeros
        scatter_within = np.zeros(shape=(num_feats, num_feats))
        scatter_between = np.zeros(shape=(num_feats, num_feats))
        unique_labels = np.unique(y)  # get all unique labels
        all_mean = np.mean(X, axis=0)  # get the mean of all the dataset

        # for all labels in the dataset get records with the selected label and update scatter within and
        # scatter between
        for label in unique_labels:
            indices = np.where(y == label)  # get the indices of the selected label
            X_label = X[indices]  # get the records
            class_mean = np.mean(X_label, axis=0)  # get mean of the records
            scatter_within += np.dot((X_label - class_mean).T, (X_label - class_mean))  # add to scatter within array
            num_records_with_label = X_label.shape[0]
            mean_diff = (class_mean - all_mean).reshape(num_feats, 1)  # get difference in means
            scatter_between += num_records_with_label * np.dot(mean_diff, mean_diff.T)  # upd

        A = np.dot(np.linalg.inv(scatter_within), scatter_between)  # get scatter_within^-1*scatter_between
        eigenvalues, eigenvectors = np.linalg.eig(A)  # get eigenvalues and eigenvectors
        eigenvectors = eigenvectors.T  # take transpose so that all eigenvectors are stored in rows
        sorted_indices = np.flip(np.argsort(eigenvalues))  # get indices in values' increasing order, then flip
        eigenvalues = eigenvalues[sorted_indices]  # store first n eigenvalues
        eigenvectors = eigenvectors[sorted_indices]  # store first n eigenvectors
        self.linear_discnants = eigenvectors[:self.num_linear_discnants]

    def transform(self, X):
        # transform the data with selected linear discriminants
        return np.dot(X, self.linear_discnants.T)
