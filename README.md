# Probabilistic Random Forest for Deep Forest (PRF4DF)

This repository contains an implementation of a Probabilistic Random Forest (PRF) designed to integrate seamlessly with the Deep Forest (gcForest) framework, particularly for handling feature uncertainties and probabilistic labels.

---

## Table of Contents

1.  [Introduction](#introduction)
2.  [Core Concepts](#core-concepts)
    * [Probabilistic Random Forest (PRF)](#probabilistic-random-forest-prf)
    * [Deep Forest (gcForest)](#deep-forest-gcforest)
    * [Feature Uncertainties ($dX$)](#feature-uncertainties-dx)
    * [Probabilistic Labels ($py$)](#probabilistic-labels-py)
3.  [How it Works](#how-it-works)
    * [DecisionTreeClassifier](#decisiontreeclassifier)
    * [RandomForestClassifier](#randomforestclassifier)
    * [SklearnCompatiblePRF](#sklearncompatibleprf)
    * [Integration with Deep Forest](#integration-with-deep-forest)
4.  [Setup and Installation](#setup-and-installation)
5.  [Usage Examples](#usage-examples)
6.  [Contributing](#contributing)

---

## 1. Introduction

This project provides a specialized implementation of a **Random Forest**, termed "**Probabilistic Random Forest**" (PRF), tailored for scenarios where input features may come with associated **uncertainties** ($dX$) and/or labels are provided as **probability distributions** ($py$) rather than single discrete values. This PRF is designed to be compatible with and plug into the `deepforest` (gcForest) framework, allowing for the construction of robust ensemble models that account for data imperfections.

**Please note:** This repository is an adaptation of the original Probabilistic Random Forest algorithm developed by Itamar Reis and Dalya Baron. The core PRF algorithm, which takes into account uncertainties in measurements (features) and assigned classes (labels) by treating them as probability distribution functions, is detailed in their paper:

[Probabilistic Random Forest: A machine learning algorithm for noisy datasets](https://arxiv.org/abs/1811.05994v1)

The original authors and their work can be found at:

* **Itamar Reis** - [https://github.com/ireis](https://github.com/ireis)
* **Dalya Baron** - [https://github.com/dalya](https://github.com/dalya)

This adaptation extends their work by providing a scikit-learn compatible wrapper (`SklearnCompatiblePRF`) that specifically facilitates its integration into the `deepforest` (gcForest) framework, along with necessary adjustments for handling meta-features and data preparation within that context.

---

## 2. Core Concepts

### Probabilistic Random Forest (PRF)

Unlike a standard Random Forest, a PRF is built to incorporate additional information during training and prediction:

* **Feature Uncertainties ($dX$):** Each feature value $X_i$ can have an associated uncertainty $dX_i$. This information can be used during the tree splitting process to make more robust decisions, especially in cases where measurements are imprecise.
* **Probabilistic Labels ($py$):** Instead of a single class label $y$, PRF can be trained with a probability distribution over classes for each sample. For example, if a sample is known to be 80% class A and 20% class B, the PRF can learn from this nuanced information.

### Deep Forest (gcForest)

**Deep Forest** (gcForest) is a tree-based ensemble method that offers an alternative to deep neural networks, often achieving comparable performance with fewer hyper-parameters and less computational cost. It builds a cascade of random forests, where the output (class probability distributions) of one layer serves as input (meta-features) for the next.

### Feature Uncertainties ($dX$)

In many real-world applications, data points are not exact. Sensor readings have noise, expert annotations have ambiguity, etc. Representing this imprecision as a $dX$ array (where $dX_{ij}$ is the uncertainty associated with feature $X_{ij}$) allows the model to make more informed decisions by considering the possible range of values for each feature.

### Probabilistic Labels ($py$)

Sometimes, the ground truth label for a sample is not a single, definite class but rather a distribution of probabilities across multiple classes. For instance, in an image classification task, an image might genuinely contain elements of both "cat" and "dog." Probabilistic labels enable the model to learn from these rich, uncertain target values.

---

## 4. How it Works

- `DecisionTreeClassifier`: This class implements a single decision tree.

- `RandomForestClassifier`: This class implements the ensemble of `DecisionTreeClassifier`.

- `SklearnCompatiblePRF`: This wrapper class is crucial for integrating the `RandomForestClassifier` into frameworks like `deepforest` that expect a scikit-learn API.

### Integration with Deep Forest

The `SklearnCompatiblePRF` class acts as an adapter. `deepforest` expects estimators for its cascade layers that implement the `fit(X, y)` and `predict_proba(X)` methods. Our `SklearnCompatiblePRF` fulfills this contract, while internally handling the complex slicing and data preparation necessary for the underlying `RandomForestClassifier` to work with optional uncertainties and probabilistic labels.

Deep Forest passes the `X_combined` array to its estimators, which is constructed by stacking:
`[Original_Features | (Optional) Original_Uncertainties | (Optional) Probabilistic_Labels | Meta_Features_from_Previous_Layer]`

The `SklearnCompatiblePRF` is responsible for correctly "un-stacking" these components to provide `X`, `dX` and `py` in the format expected by our `RandomForestClassifier`.

---

## 5. Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/carlos-vf/PRF4DF.git
    cd PRF4DF
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    .\venv\Scripts\activate # On Windows
    source venv/bin/activate # On macOS/Linux
    ```
3.  **Install necessary packages:**
    ```bash
    pip install numpy scikit-learn joblib deepforest
    ```

4.  **Ensure `PRF4DF` is a recognized package:**
    Place the `PRF4DF` directory at the root of your project or in a location where Python can find it (e.g., next to your `test.py` script).

---

## 5. Usage Examples

You can refer to the example code on the following GitHub repository for a complete picture: [PML](https://github.com/carlos-vf/PML/tree/main/test).

## 6. Contributing
We welcome contributions! Feel free to open issues to report bugs or suggest features, or submit pull requests with your improvements.