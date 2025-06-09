from deepforest import RandomForestClassifier
import numpy
from sklearn.base import BaseEstimator, ClassifierMixin
from .PRF import RandomForestClassifier


class SklearnCompatiblePRF(BaseEstimator, ClassifierMixin):
    def __init__(self, n_classes_=None, n_features_=None, use_probabilistic_labels=False, use_feature_uncertainties=False, **prf_params):
        self.prf_params = prf_params
        self.prf_model = None
        self.n_classes_ = n_classes_
        self.n_features_ = n_features_
        self.use_probabilistic_labels = use_probabilistic_labels
        self.use_feature_uncertainties = use_feature_uncertainties

    def fit(self, X, y, dX=None, sample_weight=None):
        # Your custom unpacking logic from the original file seems to handle X and X_dX (dX)
        # being passed together. To simplify and align with the framework, it's better to
        # expect X and X_dX as separate arguments. Here we adapt the existing fit method.
        
        if self.n_classes_ is None:
            self.n_classes_ = len(numpy.unique(y))
        
        n_features_for_prf = X.shape[1]

        # Initialize the PRF model
        self.prf_model = RandomForestClassifier(
            n_classes_=self.n_classes_,
            n_features_=n_features_for_prf,
            **self.prf_params
        )
        
        # Fit the underlying PRF model, passing the separated X and X_dX
        self.prf_model.fit(X=X, y=y, dX=dX)

        return self

    def predict_proba(self, X, dX=None):
        if self.prf_model is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")
            
        # Pass both X and X_dX to the underlying model's predict_proba
        proba_output = self.prf_model.predict_proba(X=X, dX=dX)
        return proba_output

    def predict(self, X, dX=None):
        probas = self.predict_proba(X, dX=dX)
        return numpy.argmax(probas, axis=1)
    

    def predict_proba_with_dX(self, X, dX=None):
        if self.prf_model is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")
        
        # Call the new method on the underlying PRF model
        mean_probas, dX_probas = self.prf_model.predict_proba_with_dX(X=X, dX=dX)
        
        return mean_probas, dX_probas
    
