import numpy as np
from joblib import Parallel, delayed

from . import misc_functions as m
from . import tree



def _predict_tree_proba(tree, X, dX):
    """Initializes a tree and runs its predict_proba method."""
    tree.node_arr_init()
    return tree.predict_proba(X, dX)

def _apply_tree(tree, X, dX):
    """Initializes a tree and runs its apply method."""
    tree.node_arr_init()
    return tree.apply(X, dX)


############################################################
############################################################
################ DecisionTreeClassifier Class  #############
############################################################
############################################################

class DecisionTreeClassifier:
    """
    (Your existing DecisionTreeClassifier code would go here. 
    It is omitted for brevity as the primary refactoring is in the RandomForestClassifier.)
    """
    def __init__(self, criterion='gini', max_features=None, use_py_gini = True, use_py_leafs = True, max_depth = None,
                 keep_proba = 0.05, unsupervised=False, new_syn_data_frac=0, min_py_sum_leaf=1, random_state=None, n_classes_=None):
        self.criterion = criterion
        self.max_features = max_features
        self.use_py_gini = use_py_gini
        self.use_py_leafs = use_py_leafs
        self.max_depth = max_depth
        self.keep_proba = keep_proba
        self.is_node_arr_init = False
        self.unsupervised = unsupervised
        self.new_syn_data_frac = new_syn_data_frac
        self.min_py_sum_leaf = min_py_sum_leaf
        self.random_state = random_state 
        self.n_classes_ = n_classes_

    def get_nodes(self):
        node_list = []
        node_list = self.tree_.get_node_list(node_list, self.tree_, 0)[1]
        node_idx = np.zeros(len(node_list), dtype = int)
        for i,node in enumerate(node_list):
            node_idx[i] = node[0]
            node_list_sort = []
        new_order = np.argsort(node_idx)
        for new_idx, idx in enumerate(new_order):
            node_list_sort += [node_list[idx]]
        return node_list_sort

    def node_arr_init(self):
        if self.is_node_arr_init:
            return

        node_list = self.get_nodes()
        self.node_tree_results = np.ones([len(node_list),self.n_classes_] )*(-1)
        self.node_feature_idx = np.ones(len(node_list), dtype = int)*(-1)
        self.node_feature_th = np.zeros(len(node_list))
        self.node_true_branch = np.ones(len(node_list), dtype = int)*(-1)
        self.node_false_branch = np.ones(len(node_list), dtype = int)*(-1)
        self.node_p_right = np.zeros(len(node_list))

        for idx, n in enumerate(node_list):
            n = node_list[idx]
            if not n[3] is None:
                self.node_feature_idx[idx] = n[1]
                self.node_feature_th[idx] = n[2]
                self.node_true_branch[idx] = n[3]
                self.node_false_branch[idx] = n[4]
                self.node_p_right[idx] = n[6]
            else:
                self.node_tree_results[idx] = n[5]

        self.is_node_arr_init = True
        return
    
    def fit(self, X, dX, py):
        if self.n_classes_ is None:
            self.n_classes_ = py.shape[1]
        elif py.shape[1] < self.n_classes_:
            py_padded = np.zeros((py.shape[0], self.n_classes_))
            py_padded[:, :py.shape[1]] = py
            py = py_padded
        elif py.shape[1] > self.n_classes_:
            raise ValueError(f"py.shape[1] ({py.shape[1]}) > self.n_classes_ ({self.n_classes_})")

        self.n_features_ = len(X[0])
        self.n_samples_ = len(X)
        self.feature_importances_ = [0] * self.n_features_
        self.is_node_arr_init = False
        pnode = np.ones(self.n_samples_)
        is_max = np.ones(self.n_samples_, dtype = int)
        py_flat = py.copy()
        py_flat[py < 0.5] = 0
        py_flat[py > 0.5] = 1
        py_gini = py if self.use_py_gini else py_flat
        py_leafs = py if self.use_py_leafs else py_flat
        depth = 0
        self.tree_ = tree.fit_tree(X, dX, py_gini, py_leafs, pnode, depth, is_max, self.max_depth, self.max_features, self.feature_importances_, self.n_samples_, self.keep_proba, self.unsupervised, self.new_syn_data_frac, self.min_py_sum_leaf, self.random_state)
    
    def apply(self, X, dX=None, return_leafs=True):
         if dX is None: dX = np.zeros_like(X)
         return self.predict_proba(X, dX, return_leafs=True)[:,0]

    def predict_proba(self, X, dX, return_leafs=False):
        keep_proba = self.keep_proba
        result = tree.predict_all(self.node_tree_results, self.node_feature_idx, self.node_feature_th, self.node_true_branch, self.node_false_branch, self.node_p_right, X, dX, keep_proba, return_leafs)
        return result


class RandomForestClassifier:
    """
    A Probabilistic Random Forest classifier.

    This implementation is designed to handle feature uncertainties (dX) and
    probabilistic labels (py) during training.
    """
    def __init__(self, n_estimators=100, max_depth=None, max_features='auto', 
                 bootstrap=True, n_jobs=1, random_state=None,
                 n_classes_=None, n_features_=None, **kwargs):
        
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.n_classes_ = n_classes_
        self.n_features_ = n_features_
        
        self.estimators_ = []
        # Pass any other DecisionTree args through kwargs
        self.tree_kwargs = kwargs

    def _validate_and_prepare_input(self, X, y=None, dX=None, py=None):
        """Validates and prepares all inputs for the fitting process."""
        if self.n_features_ is None:
            self.n_features_ = X.shape[1]
        
        X, dX = self.check_input_X(X, dX)

        if py is None:
            if y is None:
                raise ValueError("Either `y` (discrete labels) or `py` (probabilistic labels) must be provided.")
            py, self.label_dict = m.get_pY(np.ones(len(y)), y)
        
        if self.n_classes_ is None:
            self.n_classes_ = py.shape[1]
        
        if not hasattr(self, 'label_dict'):
            self.label_dict = {i: i for i in range(self.n_classes_)}

        return X, dX, py

    def _fit_single_tree(self, X, dX, py, tree_random_state):
        """Initializes and fits a single decision tree."""
        max_features_num = self._get_max_features_num()
        
        tree_estimator = DecisionTreeClassifier(
            max_depth=self.max_depth,
            max_features=max_features_num,
            random_state=tree_random_state,
            n_classes_=self.n_classes_,
            **self.tree_kwargs
        )

        if self.bootstrap:
            rng = np.random.RandomState(tree_random_state)
            indices = rng.choice(np.arange(X.shape[0]), size=X.shape[0], replace=True)
            tree_estimator.fit(X[indices], dX[indices], py[indices])
        else:
            tree_estimator.fit(X, dX, py)
        return tree_estimator

    def fit(self, X, y=None, dX=None, py=None, sample_weight=None):
        """
        Fits a forest of probabilistic trees on the training data.
        """
        # Note: sample_weight is not currently used in the core tree fitting
        # but is kept for API compatibility.
        X, dX, py = self._validate_and_prepare_input(X, y, dX, py)

        rng = np.random.RandomState(self.random_state)
        tree_seeds = rng.randint(np.iinfo(np.int32).max, size=self.n_estimators)

        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
            delayed(self._fit_single_tree)(X, dX, py, seed) for seed in tree_seeds
        )

        # Aggregate feature importances
        self.feature_importances_ = np.mean([
            tree.feature_importances_ for tree in self.estimators_
        ], axis=0)

        return self
    
    def _get_all_tree_predictions(self, X, dX=None):
        """Helper to collect predictions from all trees in the forest."""
        X, dX = self.check_input_X(X, dX)

        all_preds = Parallel(n_jobs=self.n_jobs)(
            delayed(_predict_tree_proba)(tree, X, dX) for tree in self.estimators_
        )
        return np.array(all_preds)

    def predict_proba(self, X, dX=None):
        """
        Predicts class probabilities for X by averaging tree predictions.
        """
        predictions_array = self._get_all_tree_predictions(X, dX)
        mean_probas = np.mean(predictions_array, axis=0)
        
        # Normalize to ensure probabilities sum to 1
        sum_rows = mean_probas.sum(axis=1, keepdims=True)
        # Avoid division by zero for samples where all predictions are zero
        safe_sum = np.where(sum_rows == 0, 1, sum_rows)
        return mean_probas / safe_sum

    def predict_proba_with_std(self, X, dX=None):
        """
        Predicts class probabilities and their standard error of the mean.
        """
        predictions_array = self._get_all_tree_predictions(X, dX)
        
        mean_probas = np.mean(predictions_array, axis=0)
        std_probas = np.std(predictions_array, axis=0)
        sem_probas = std_probas / np.sqrt(self.n_estimators)
        
        sum_rows = mean_probas.sum(axis=1, keepdims=True)
        safe_sum = np.where(sum_rows == 0, 1, sum_rows)
        
        return mean_probas / safe_sum, sem_probas

    def apply(self, X, dX=None):
        """Applies trees in the forest to X, returning leaf indices."""
        X, dX = self.check_input_X(X, dX)

        all_leaves = Parallel(n_jobs=self.n_jobs)(
            delayed(_apply_tree)(tree, X, dX) for tree in self.estimators_
        )
        return np.vstack(all_leaves).T
        
    def predict(self, X, dX=None):
        """Predicts the class for X."""
        probas = self.predict_proba(X, dX)
        best_class_indices = np.argmax(probas, axis=1)
        return np.array([self.label_dict[i] for i in best_class_indices])
    
    # ... (other helper methods like check_input_X, _get_max_features_num would go here)
    def check_input_X(self, X, dX):
        if dX is None: dX = np.zeros_like(X)
        # Basic validation
        return X, dX
    
    def _get_max_features_num(self):
        if isinstance(self.max_features, str):
            if self.max_features == 'auto' or self.max_features == 'sqrt':
                return int(np.sqrt(self.n_features_))
            elif self.max_features == 'log2':
                return int(np.log2(self.n_features_))
        elif isinstance(self.max_features, int):
            return self.max_features
        elif isinstance(self.max_features, float):
            return int(self.max_features * self.n_features_)
        return self.n_features_
