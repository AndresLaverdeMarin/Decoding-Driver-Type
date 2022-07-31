# -*- coding: utf-8 -*-
from sklearn.svm import LinearSVC
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
        Linear SVM model.
    """

    clf = OneVsRestClassifier(
        BaggingClassifier(
            LinearSVC(verbose=1),
            max_samples=1.0 / n_estimators,
            n_estimators=n_estimators,
        )
    )

    return clf
