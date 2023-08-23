import numpy as np
import os

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from games.fashionmnist.utils import mnist_reader

class FashionMnist:
    def __init__(
            self, path=os.path.join(os.path.dirname(__file__), 'fashionmnist', 'data', 'fashion')):
        self.path = path
        self.standard_scaler = StandardScaler()

    def load_data(self):
        X_train, y_train = mnist_reader.load_mnist(self.path, kind='train')
        X_test, y_test = mnist_reader.load_mnist(self.path, kind='t10k')

        X_train, X_test = X_train / 255, X_test / 255
        self.standard_scaler.fit(X_train)
        return X_train, y_train, X_test, y_test

    def preprocess_train_and_test_data(self, X_train, X_test, n_components):
        X_train_reduced = self.apply_pca(X_train, n_components=n_components)
        X_test_reduced = self.apply_pca(X_test, n_components=n_components)
        X_train_normalized, X_test_normalized = self.normalize_data(X_train_reduced, X_test_reduced)
        return X_train_normalized, X_test_normalized

    def apply_pca(self, data, n_components):
        standardized_data = self.standard_scaler.transform(data)

        pca = PCA(n_components=n_components)
        transformed_data = pca.fit_transform(standardized_data)
        return transformed_data
    
    def normalize_data(self, X_train, X_test):
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        return X_train, X_test

    def filter_by_label(self, X_data, y_data, labels):
        if type(labels) is int:
            labels = [labels]
        for n, label in enumerate(labels):
            mask = y_data==label
            X_filtered = X_data[mask]
            y_filtered = y_data[mask]
            if n == 0:
                X_data_out = X_filtered
                y_data_out = y_filtered
            else:
                X_data_out = np.vstack((X_data_out, X_filtered))
                y_data_out = np.hstack((y_data_out, y_filtered))
        return X_data_out, y_data_out
