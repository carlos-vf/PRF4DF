import numpy
from joblib import Parallel, delayed
import threading
from sklearn.base import BaseEstimator, ClassifierMixin

from . import misc_functions as m
from . import tree

############################################################
############################################################
################ DecisionTreeClassifier Class  #############
############################################################
############################################################

class DecisionTreeClassifier:
    """
    This the the decision tree classifier class.
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
        node_idx = numpy.zeros(len(node_list), dtype = int)
        for i,node in enumerate(node_list):
            node_idx[i] = node[0]
            node_list_sort = []
        new_order = numpy.argsort(node_idx)
        for new_idx, idx in enumerate(new_order):
            node_list_sort += [node_list[idx]]
        return node_list_sort

    def node_arr_init(self):
        if self.is_node_arr_init:
            return

        node_list = self.get_nodes()

        node_tree_results = numpy.ones([len(node_list),self.n_classes_] )*(-1)
        node_feature_idx = numpy.ones(len(node_list), dtype = int)*(-1)
        node_feature_th = numpy.zeros(len(node_list))
        node_true_branch = numpy.ones(len(node_list), dtype = int)*(-1)
        node_false_branch = numpy.ones(len(node_list), dtype = int)*(-1)
        node_p_right = numpy.zeros(len(node_list))

        for idx, n in enumerate(node_list):
            n = node_list[idx]
            if not n[3] is None:
                node_feature_idx[idx] = n[1]
                node_feature_th[idx] = n[2]
                node_true_branch[idx] = n[3]
                node_false_branch[idx] = n[4]
                node_p_right[idx] = n[6]
            else:
                node_tree_results[idx] = n[5]

        self.node_feature_idx = node_feature_idx
        self.node_feature_th = node_feature_th
        self.node_true_branch = node_true_branch
        self.node_false_branch = node_false_branch
        self.node_tree_results = node_tree_results
        self.node_p_right = node_p_right
        self.is_node_arr_init = True
        return

    def fit(self, X, pX, py):
        """
        the DecisionTreeClassifier.fit() function with a similar appearance to that of sklearn
        """
        # Ensure n_classes_ is set, either from init or from py.shape[1]
        if self.n_classes_ is None:
            self.n_classes_ = py.shape[1]
        # Pad py if the current fold's classes are fewer than the global n_classes_
        elif py.shape[1] < self.n_classes_:
            py_padded = numpy.zeros((py.shape[0], self.n_classes_))
            py_padded[:, :py.shape[1]] = py
            py = py_padded
        # Raise error if py has more classes than expected (should not happen with correct setup)
        elif py.shape[1] > self.n_classes_:
            raise ValueError(f"py.shape[1] ({py.shape[1]}) is greater than self.n_classes_ ({self.n_classes_}). This indicates a class mismatch.")

        self.n_features_ = len(X[0])
        self.n_samples_ = len(X)
        self.feature_importances_ = [0] * self.n_features_
        self.is_node_arr_init = False

        pnode = numpy.ones(self.n_samples_)
        is_max = numpy.ones(self.n_samples_, dtype = int)

        py_flat = py.copy()
        py_flat[py < 0.5] = 0
        py_flat[py > 0.5] = 1

        if self.use_py_gini:
            py_gini = py
        else:
            py_gini = py_flat

        if self.use_py_leafs:
            py_leafs = py
        else:
            py_leafs = py_flat
        depth = 0

        self.tree_ = tree.fit_tree(X, pX, py_gini, py_leafs, pnode, depth, is_max, self.max_depth, self.max_features, self.feature_importances_, self.n_samples_, self.keep_proba, self.unsupervised, self.new_syn_data_frac, self.min_py_sum_leaf, self.random_state)

    def predict_proba(self, X, dX, return_leafs=False):
        """
        The DecisionTreeClassifier.predict_proba() function with a similar appearance to the of sklearn
        """
        keep_proba = self.keep_proba
        result = tree.predict_all(self.node_tree_results, self.node_feature_idx, self.node_feature_th, self.node_true_branch, self.node_false_branch, self.node_p_right, X, dX, keep_proba, return_leafs)
        return result


############################################################
############################################################
################ RandomForestClassifier Class  #############
############################################################
############################################################

class RandomForestClassifier:
    def __init__(self, n_estimators=10, criterion='gini', max_features='auto', use_py_gini = True, use_py_leafs = True,
                 max_depth = None, keep_proba = 0.05, bootstrap=True, new_syn_data_frac=0, min_py_sum_leaf=1, n_jobs=1, random_state=None, n_classes_=None):
        self.n_estimators_ = n_estimators
        self.criterion = criterion
        self.max_features = max_features
        self.estimators_ = []
        self.use_py_gini = use_py_gini
        self.use_py_leafs = use_py_leafs
        self.max_depth = max_depth
        self.keep_proba = keep_proba
        self.bootstrap = bootstrap
        self.new_syn_data_frac = new_syn_data_frac
        self.min_py_sum_leaf = min_py_sum_leaf
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.n_classes_ = n_classes_

    def check_input_X(self, X, dX):
        # Ensure X is float and writable (create a copy if necessary)
        if not numpy.issubdtype(X.dtype, numpy.floating) or not X.flags['WRITEABLE']:
            X = X.astype(numpy.float64, copy=True) 
        
        if dX is None:
            # If X is 1D, reshape X.shape to (len(X), 1) to make dX 2D
            if X.ndim == 1:
                dX = numpy.zeros((X.shape[0], 1))
            else:
                dX = numpy.zeros(X.shape)
        
        # Ensure dX is float and writable (create a copy if necessary)
        if not numpy.issubdtype(dX.dtype, numpy.floating) or not dX.flags['WRITEABLE']:
            dX = dX.astype(numpy.float64, copy=True)

        dX[numpy.isnan(dX)] = 0
        X[numpy.isinf(dX)] = numpy.nan

        return X, dX

    def _choose_objects(self, X, pX, py):
        """
        function builds a sample of the same size as the input data, but chooses the objects with replacement
        according to the given probability matrix
        """
        if self.random_state is not None:
            numpy.random.seed(self.random_state)

        nof_objects = py.shape[0]
        objects_indices = numpy.arange(nof_objects)
        objects_chosen = numpy.random.choice(objects_indices, nof_objects, replace=True)
        X_chosen = X[objects_chosen, :]
        pX_chosen = pX[objects_chosen, :]
        py_chosen = py[objects_chosen, :]

        return X_chosen, pX_chosen, py_chosen

    def _fit_single_tree(self, X, pX, py):
        tree = DecisionTreeClassifier(criterion=self.criterion,
                                      max_features=self.max_features_num,
                                      use_py_gini = self.use_py_gini,
                                      use_py_leafs = self.use_py_leafs,
                                      max_depth = self.max_depth,
                                      keep_proba = self.keep_proba,
                                      unsupervised = self.unsupervised,
                                      new_syn_data_frac = self.new_syn_data_frac,
                                      min_py_sum_leaf=self.min_py_sum_leaf,
                                      random_state=self.random_state,
                                      n_classes_=self.n_classes_) # Pass RF's n_classes_ to DT

        if self.bootstrap:
            X_chosen, pX_chosen, py_chosen = self._choose_objects(X, pX, py)
            tree.fit(X_chosen, pX_chosen, py_chosen)
        else:
            tree.fit(X, pX, py)
        return tree

    def fit(self, X, dX=None, y=None, py=None):
        """
        The RandomForestClassifier.fit() function with a similar appearance to that of sklearn
        """
        n_features = X.shape[1]
        n_objects = X.shape[0]
        self.n_features_ = n_features
        self.feature_importances_ = numpy.zeros(self.n_features_)

        if self.max_features == 'auto' or self.max_features == 'sqrt':
            self.max_features_num = int(numpy.sqrt(self.n_features_))
        elif self.max_features == 'log2':
            self.max_features_num = int(numpy.log2(self.n_features_))
        elif type(self.max_features) == int:
            self.max_features_num = self.max_features
        elif type(self.max_features) == float:
            self.max_features_num = int(self.max_features * self.n_features_)
        else:
            self.max_features_num = self.n_features_

        self.unsupervised = False
        # Only determine n_classes_ if it's not already set (i.e., passed from __init__)
        if self.n_classes_ is None:
            if py is not None:
                self.n_classes_ = py.shape[1]
                self.label_dict = {i:i for i in range(self.n_classes_)}
            elif y is not None:
                # If py is NOT provided, but y is, convert y to py internally.
                py, label_dict = m.get_pY(numpy.ones(len(y)), y)
                self.n_classes_ = py.shape[1]
                self.label_dict = label_dict
            else:
                self.unsupervised = True
                py = numpy.zeros([n_objects,2])
                py[:,0] = 1
                self.n_classes_ = 2
                self.label_dict = {i:i for i in range(self.n_classes_)}
        # If n_classes_ was set in __init__, and py is still None,
        # but y is provided, we still need to create py based on the global n_classes_.
        elif y is not None and py is None:
            # Ensure m.get_pY can correctly map y to indices for self.n_classes_ columns
            # This implicitly means y should already be 0-indexed for the global class range
            py, label_dict = m.get_pY(numpy.ones(len(y)), y)
            # If the py generated is smaller than self.n_classes_, pad it.
            if py.shape[1] < self.n_classes_:
                py_padded = numpy.zeros((py.shape[0], self.n_classes_))
                py_padded[:, :py.shape[1]] = py
                py = py_padded
            elif py.shape[1] > self.n_classes_:
                raise ValueError(f"In RandomForestClassifier.fit: py.shape[1] ({py.shape[1]}) is greater than self.n_classes_ ({self.n_classes_}). This implies a mismatch in class definition.")
            self.label_dict = label_dict # Update label_dict from this conversion

        X, dX = self.check_input_X(X, dX)

        if self.n_jobs == 1:
            tree_list = [self._fit_single_tree(X, dX, py) for i in range(self.n_estimators_)]
        else:
            tree_list = Parallel(n_jobs=self.n_jobs, verbose = 0)(delayed(self._fit_single_tree)
                                                                 (X, dX, py)        for i in range(self.n_estimators_))
        self.estimators_ = []
        for tree in tree_list:
            self.estimators_.append(tree)
            self.feature_importances_ += numpy.array(tree.feature_importances_)

        self.feature_importances_ /= self.n_estimators_
        return self

    def predict_single_object(self, row):
        # let all the trees vote
        summed_probabilites = numpy.zeros(self.n_classes_)
        for tree_index in range(0, self.n_estimators_):
            y_proba_tree = self.estimators_[tree_index].predict_proba_one_object(row)
            summed_probabilites += y_proba_tree
        return numpy.argmax(summed_probabilites)

    def pick_best(self, class_proba):
        n_objects = class_proba.shape[0]
        new_class_proba = numpy.zeros(class_proba.shape)
        best_idx = numpy.argmax(class_proba, axis = 1)
        for i in range(n_objects):
            new_class_proba[i,best_idx[i]] = class_proba[i,best_idx[i]]
        return new_class_proba

    def predict_single_tree(self, predict, X, dX, out):
        prediction = predict(X,dX)
        if len(out) == 1:
            out[0] += prediction
        else:
            for i in range(len(out)):
                out[i] += prediction[i]

    def predict_proba(self, X, dX=None):
        """
        The RandomForestClassifier.predict() function with a similar appearance to that of sklearn
        """
        proba = numpy.zeros((X.shape[0], self.n_classes_), dtype=numpy.float64)
        X, dX = self.check_input_X(X, dX)

        for i, tree_estimator in enumerate(self.estimators_):
            tree_estimator.node_arr_init()
            single_tree_prediction = tree_estimator.predict_proba(X, dX)  
            proba += single_tree_prediction

        sum_rows = proba.sum(axis=1, keepdims=True)
        proba = numpy.where(sum_rows == 0, 0, proba / sum_rows)
        return proba 

    def apply(self, X, dX=None):
        X, dX = self.check_input_X(X, dX)
        dX = numpy.zeros(X.shape) # TODO: return leafs with probabilities for PRF
        for i, tree in enumerate(self.estimators_):
            tree.node_arr_init()
        leafs = [tree.predict_proba(X, dX, return_leafs=True)[:,0].reshape(-1,1) for tree in self.estimators_]
        leafs = numpy.hstack(leafs).astype(int)
        return leafs

    def predict(self, X, dX=None, return_leafs=False):
        y_pred_inds = numpy.argmax(self.predict_proba(X, dX), axis = 1)
        y_pred = numpy.array([self.label_dict[i] for i in y_pred_inds])
        return y_pred

    def score(self, X, y, dX=None):
        y_pred = self.predict(X, dX)
        score = (y_pred == (y)).sum()/len(y)
        return score

    def __str__(self):
        sb = []
        do_not_print = ['estimators_', 'use_py_gini', 'use_py_leafs', 'label_dict', 'new_syn_data_frac']
        for key in self.__dict__:
            if key not in do_not_print:
                sb.append("{key}='{value}'".format(key=key, value=self.__dict__[key]))
        sb = 'ProbabilisticRandomForestClassifier(' + ', '.join(sb) + ')'
        return sb

    def __repr__(self):
        return self.__str__()
    
class SklearnCompatiblePRF(BaseEstimator, ClassifierMixin):
    # Added n_classes_ as an explicit argument to __init__
    def __init__(self, n_classes_=None, **prf_params):
        self.prf_params = prf_params
        self.prf_model = None
        self.n_classes_ = n_classes_ # Store the global n_classes_

    def fit(self, X, y):
        # Ensure n_classes_ was provided during initialization
        if self.n_classes_ is None:
            raise ValueError("SklearnCompatiblePRF must be initialized with the total number of classes (n_classes_=...).")

        # Pass the global n_classes_ to the RandomForestClassifier's constructor
        self.prf_model = RandomForestClassifier(n_classes_=self.n_classes_, **self.prf_params)

        dX_for_prf = numpy.zeros(X.shape) 
        
        # Pass the raw y (0-indexed for the fold's classes) directly to RandomForestClassifier.fit.
        # RandomForestClassifier's fit method will handle the conversion to py based on its fixed n_classes_.
        self.prf_model.fit(X, dX_for_prf, y=y) 
        
        return self
    
    def predict_proba(self, X):
        dX_for_prf = numpy.zeros(X.shape)
        return self.prf_model.predict_proba(X, dX=dX_for_prf)

    def predict(self, X):
        probas = self.predict_proba(X)
        # DeepForest (or other scikit-learn tools) might expect 0-indexed labels
        # corresponding to the class labels passed in y_train.
        return numpy.argmax(probas, axis=1)