# -*- coding: utf-8 -*-
"""
Logistic Regression Classifier with Bagging Ensemble

This module implements a Logistic Regression classifier with bagging ensemble
for driver type identification. The model combines the interpretability of
logistic regression with the robustness of ensemble methods.

Logistic Regression provides probabilistic predictions and is particularly
useful as a baseline model for binary and multi-class classification tasks.
The bagging approach improves stability and reduces overfitting.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import BaggingClassifier


def model(n_estimators=10):
    """
    Build a Logistic Regression classifier with bagging ensemble.

    This function creates a robust probabilistic classifier by combining multiple
    Logistic Regression estimators through bootstrap aggregating (bagging). Each
    base estimator is trained on a random subset of the training data, and final
    predictions are made by averaging probability estimates across all estimators.

    The One-vs-Rest strategy enables multi-class classification by training one
    binary logistic regression model per class, treating it as positive while
    all other classes are treated as negative.

    Parameters
    ----------
    n_estimators : int, optional
        Number of base estimators in the bagging ensemble (default: 10).
        Increasing this value improves model stability and generalization
        at the cost of higher computational requirements.
        Each estimator trains on (1/n_estimators) fraction of the data.

    Returns
    -------
    OneVsRestClassifier
        A configured Logistic Regression classifier with bagging ensemble,
        ready for training using the fit() method. The model supports both
        predict() for class labels and predict_proba() for class probabilities.

    Notes
    -----
    - Logistic Regression uses the logit function for probabilistic predictions
    - Bagging reduces model variance and improves generalization
    - Each base estimator is trained on 1/n_estimators of the training samples
    - Verbose output is enabled to monitor training progress
    - The model inherently provides feature importance through coefficients

    Examples
    --------
    >>> clf = model(n_estimators=10)
    >>> clf.fit(X_train, y_train)
    >>> predictions = clf.predict(X_test)
    >>> probabilities = clf.predict_proba(X_test)

    See Also
    --------
    SVM_linear : Alternative linear classifier using support vector machines
    """
    # Create ensemble classifier with One-vs-Rest multi-class strategy
    # and bootstrap aggregating for improved robustness
    clf = OneVsRestClassifier(
        BaggingClassifier(
            LogisticRegression(verbose=1),  # Base estimator with progress output
            max_samples=1.0 / n_estimators,  # Fraction of samples per estimator
            n_estimators=n_estimators,  # Number of estimators in ensemble
        )
    )

    return clf
