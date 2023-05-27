import numpy as np
from scipy.linalg import eigh
from sklearn.base import BaseEstimator, TransformerMixin

class CSP(BaseEstimator, TransformerMixin):
    def __init__(self, n_components):
        self.n_components = n_components
        self.filters_ = None
    
    def fit(self, X, y):
        class_covs = self._compute_class_covariance(X, y)
        evals, evecs = eigh(class_covs[0], class_covs[0] + class_covs[1])
        
        # Tri des valeurs propres et vecteurs propres dans l'ordre décroissant des valeurs propres
        sort_indices = np.argsort(evals)[::-1]
        evals = evals[sort_indices]
        evecs = evecs[:, sort_indices]
        
        # Sélection des composants CSP
        self.filters_ = np.dot(np.sqrt(np.linalg.inv(np.diag(np.log(evals[:self.n_components]))))), evecs.T[:self.n_components]
        
        return self
    
    def transform(self, X):
        # Projection des données sur les composants CSP
        X_csp = np.dot(X, self.filters_.T)
        
        return X_csp
    
    def _compute_class_covariance(self, X, y):
        _, n_channels, _ = X.shape
        class_covs = []
        classes = np.unique(y)
        
        for class_label in classes:
            class_data = X[y == class_label]
            class_data = np.transpose(class_data, [1, 0, 2])
            class_data = class_data.reshape(n_channels, -1)
            class_cov = np.cov(class_data, rowvar=False)
            class_covs.append(class_cov)
        
        return class_covs
