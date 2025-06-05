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

    def fit(self, X_combined, y_discrete_from_deepforest):
        if self.n_classes_ is None:
            raise ValueError("SklearnCompatiblePRF must be initialized with the total number of classes (n_classes_=...).")
        if self.n_features_ is None:
            raise ValueError("SklearnCompatiblePRF must be initialized with n_features_.")

        n_samples = X_combined.shape[0]
        n_X_orig_features_cols = self.n_features_

        # Use a running index to dynamically slice X_combined
        current_column_index = 0

        # Extract X_orig_features
        X_orig_features = X_combined[:, current_column_index : n_X_orig_features_cols]
        current_column_index += n_X_orig_features_cols

        # Extract dX_orig_uncertainties (if used)
        dX_orig_uncertainties = None
        if self.use_feature_uncertainties:
            n_dX_orig_uncertainties_cols = self.n_features_
            dX_orig_uncertainties = X_combined[:, current_column_index : current_column_index + n_dX_orig_uncertainties_cols]
            current_column_index += n_dX_orig_uncertainties_cols
        else:
            dX_orig_uncertainties = numpy.zeros((n_samples, n_X_orig_features_cols), dtype=numpy.float64)

        # Extract py_for_prf (if used)
        py_for_prf = None
        if self.use_probabilistic_labels:
            n_py_cols = self.n_classes_
            py_for_prf = X_combined[:, current_column_index : current_column_index + n_py_cols]
            current_column_index += n_py_cols

        # DeepForest Meta-Features (remaining columns)
        meta_features_from_deepforest = X_combined[:, current_column_index:]

        # Create dX_meta_features (always zeros, regardless of meta-features presence)
        if meta_features_from_deepforest.shape[1] > 0:
            dX_meta_features = numpy.zeros_like(meta_features_from_deepforest, dtype=numpy.float64)
        else:
            dX_meta_features = numpy.zeros((n_samples, 0), dtype=numpy.float64)

        # Final concatenation for PRF's X and dX inputs
        X_final_for_prf = numpy.hstack([X_orig_features, meta_features_from_deepforest])
        dX_final_for_prf = numpy.hstack([dX_orig_uncertainties, dX_meta_features])

        n_effective_features_for_prf = X_final_for_prf.shape[1]

        # Initialize the PRF model
        self.prf_model = RandomForestClassifier(
            n_classes_=self.n_classes_,
            n_features_=n_effective_features_for_prf,
            **self.prf_params
        )

        # Conditional call to prf_model.fit based on whether probabilistic labels are used
        if self.use_probabilistic_labels:
            if py_for_prf is None:
                 raise ValueError("use_probabilistic_labels is True but py_for_prf was not extracted.")
            self.prf_model.fit(X=X_final_for_prf, dX=dX_final_for_prf, py=py_for_prf)
        else:
            self.prf_model.fit(X=X_final_for_prf, dX=dX_final_for_prf, y=y_discrete_from_deepforest)

        return self

    def predict_proba(self, X_combined):
        if self.n_features_ is None:
            raise ValueError("SklearnCompatiblePRF must be initialized with n_features_ for prediction.")
        if self.prf_model is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")

        n_samples = X_combined.shape[0]

        # dynamic slicing logic
        current_column_index = 0

        X_orig_features = X_combined[:, current_column_index : self.n_features_]
        current_column_index += self.n_features_

        # Extract dX_orig_uncertainties if used
        if self.use_feature_uncertainties:
            dX_orig_uncertainties = X_combined[:, current_column_index : current_column_index + self.n_features_]
            current_column_index += self.n_features_
        else:
            dX_orig_uncertainties = numpy.zeros((n_samples, self.n_features_), dtype=numpy.float64)

        # Extract py_for_prf if used
        if self.use_probabilistic_labels:
            current_column_index += self.n_classes_
        
        meta_features_from_deepforest = X_combined[:, current_column_index:]
        
        # Create dX_meta_features (always zeros, regardless of meta-features presence)
        if meta_features_from_deepforest.shape[1] > 0:
            dX_meta_features = numpy.zeros_like(meta_features_from_deepforest, dtype=numpy.float64)
        else:
            dX_meta_features = numpy.zeros((n_samples, 0), dtype=numpy.float64)

        X_final_for_prf = numpy.hstack([X_orig_features, meta_features_from_deepforest])
        dX_final_for_prf = numpy.hstack([dX_orig_uncertainties, dX_meta_features])

        proba_output = self.prf_model.predict_proba(X=X_final_for_prf, dX=dX_final_for_prf)

        return proba_output

    def predict(self, X_combined):
        probas = self.predict_proba(X_combined)
        return numpy.argmax(probas, axis=1)