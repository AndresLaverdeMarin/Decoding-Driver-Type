# -*- coding: utf-8 -*-
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import BaggingClassifier


def model(n_estimators=10):
    """
    Parameters
    ----------
    n_estimators: int
        The number of base estimators in the ensemble.

    Returns
    -------
        Logistic regression model.
    """

    clf = OneVsRestClassifier(
        BaggingClassifier(
            LogisticRegression(verbose=1),
            max_samples=1.0 / n_estimators,
            n_estimators=n_estimators,
        )
    )

    return clf