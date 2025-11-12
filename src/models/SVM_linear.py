# -*- coding: utf-8 -*-
"""
Linear Support Vector Machine (SVM) Classifier

This module implements a Linear SVM classifier with bagging ensemble for driver
type identification. The model uses a One-vs-Rest strategy for multi-class
classification combined with bootstrap aggregating for improved robustness.

The Linear SVM is particularly effective for high-dimensional feature spaces
and provides a computationally efficient alternative to kernel-based SVMs.
"""

from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import BaggingClassifier


def model(n_estimators=10):
    """
    Build a Linear SVM classifier with bagging ensemble.

    This function creates a robust classifier by combining multiple Linear SVM
    estimators through bootstrap aggregating (bagging). Each base estimator is
    trained on a random subset of the training data, and predictions are made
    by majority voting across all estimators.

    The One-vs-Rest strategy handles multi-class classification by training
    one binary classifier per class, which distinguishes that class from all
    other classes combined.

    Parameters
    ----------
    n_estimators : int, optional
        Number of base estimators in the bagging ensemble (default: 10).
        Higher values increase robustness but also computational cost.
        Each estimator is trained on (1/n_estimators) fraction of the data.

    Returns
    -------
    OneVsRestClassifier
        A configured Linear SVM classifier with bagging ensemble, ready for
        training using the fit() method.

    Notes
    -----
    - LinearSVC uses the liblinear library for efficient linear classification
    - Bagging reduces variance and helps prevent overfitting
    - Each base estimator sees 1/n_estimators of the training samples
    - Verbose output is enabled for monitoring training progress

    Examples
    --------
    >>> clf = model(n_estimators=10)
    >>> clf.fit(X_train, y_train)
    >>> predictions = clf.predict(X_test)
    """
    # Create ensemble classifier with One-vs-Rest multi-class strategy
    # and bootstrap aggregating for improved generalization
    clf = OneVsRestClassifier(
        BaggingClassifier(
            LinearSVC(verbose=1),  # Base estimator with progress output
            max_samples=1.0 / n_estimators,  # Fraction of data per estimator
            n_estimators=n_estimators,  # Number of estimators in ensemble
        )
    )

    return clf
