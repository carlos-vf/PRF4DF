from sklearn.base import BaseEstimator, ClassifierMixin
from .PRF import RandomForestClassifier


class SklearnCompatiblePRF(BaseEstimator, ClassifierMixin):
    """
    A scikit-learn compatible wrapper for the Probabilistic Random Forest.
    
    This class allows the PRF to be used seamlessly within scikit-learn
    pipelines and the deep-forest framework.
    """
    def __init__(self, n_estimators=100, max_depth=None, max_features='auto',
                 n_jobs=1, random_state=None, n_classes_=None, n_features_=None, **kwargs):
        
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.n_classes_ = n_classes_
        self.n_features_ = n_features_
        self.kwargs = kwargs
        
        self.prf_model = None

    def fit(self, X, y, dX=None, py=None, sample_weight=None):
        """Instantiates and fits the core PRF model."""
        self.prf_model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            max_features=self.max_features,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            n_classes_=self.n_classes_,
            n_features_=self.n_features_,
            **self.kwargs
        )
        self.prf_model.fit(X=X, y=y, dX=dX, py=py, sample_weight=sample_weight)
        return self

    def predict(self, X, dX=None):
        return self.prf_model.predict(X, dX=dX)

    def predict_proba(self, X, dX=None):
        return self.prf_model.predict_proba(X, dX=dX)

    def predict_proba_with_std(self, X, dX=None):
        return self.prf_model.predict_proba_with_std(X, dX=dX)