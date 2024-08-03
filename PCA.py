import numpy as np


class PCA:
    def __init__(self, num_princ_comps):
        self.num_princ_comps = num_princ_comps

    def fit_transform(self, X):
        cov = np.cov(X.T)  # np.cov needs samples as columns. we take X.T because in our dataset
        # samples are originally rows
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        eigenvectors = eigenvectors.T  # taking transpose so that eigenvectors are stored inside rows instead of columns
        sorted_indices = np.flip(np.argsort(eigenvalues))  # get indices in values' increasing order, then flip
        eigenvectors = eigenvectors[sorted_indices, :]  # get eigenvectors sorted by highest eigenvalues
        # select first n eigenvectors as principal components
        selected_princ_comps = eigenvectors[: self.num_princ_comps]
        return np.dot(X, selected_princ_comps.T)  # project the selected prime components onto the dataset and return
