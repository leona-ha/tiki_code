from sklearn.base import BaseEstimator, RegressorMixin
from merf.merf import MERF

class MERFWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, gll_early_stop_threshold=0.01, max_iterations=20):
        self.gll_early_stop_threshold = gll_early_stop_threshold
        self.max_iterations = max_iterations
        self.merf_model = MERF(gll_early_stop_threshold=gll_early_stop_threshold, max_iterations=max_iterations)
        self.Z_ = None
        self.clusters_ = None

    def fit(self, X, y, Z, clusters):
        self.Z_ = Z
        self.clusters_ = clusters
        self.merf_model.fit(X, Z, clusters, y)
        return self

        return self

    def predict(self, X, Z=None, clusters=None):
        if Z is None:
            Z = self.Z_
        if clusters is None:
            clusters = self.clusters_
    
        # Check alignment
        if len(Z) != len(X) or len(clusters) != len(X):
            raise ValueError(f"Shape mismatch: X={len(X)}, Z={len(Z)}, clusters={len(clusters)}")
    
        return self.merf_model.predict(X, Z, clusters)



    def get_params(self, deep=True):
        return {
            "gll_early_stop_threshold": self.gll_early_stop_threshold,
            "max_iterations": self.max_iterations,
        }

    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        return self
