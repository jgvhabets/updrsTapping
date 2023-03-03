"""
Perform Cross-Validation in prediction model's
development phase.
"""

# import public packages
import numpy as np
from pandas import crosstab
from itertools import product

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


def get_cvFold_predictions_dicts(
    X_cv,
    y_cv,
    cv_method=StratifiedKFold,
    n_folds: int=4,
    clf=LDA(),
    sv_kernel='linear',
    lr_solver='lbfgs',
    random_state=27,
    verbose=True,
):
    """
    Calculates true labels, predicted scores (with
    default .5 threshold), and predicted probabilities
    per fold.as_integer_ratio
    Input:
        - X_cv: X matrix used in total for crossvalidation
        - y_cv: corresponding y vector
        - cv_method: must be imported crossvalidation function,
            will be called within function with n_folds
        - n_folds: number of folds to use
        - clf: must be imported classifier function,
            needs to be inserted with parameter definition
            within parenthesis
    
    Returns:
        - dict with predicted scores,
        - dict with predicted probabilities
        - dict with true labels
        - dict with the indices of the resulting y-values
            in the original X_cv and y_cv (used to take sample
            with positive or negative preditions in next
            classifier)

    """
    # check and define Clf inserted as string
    if type(clf) == str:
        if clf.lower() == 'lda':
            clf = LDA()
        elif clf.lower() == 'logreg':
            clf = LogisticRegression(
                random_state=random_state,
                solver=lr_solver,
            )
        elif clf.lower() == 'svm' or clf.lower() == 'svc':
            clf = SVC(
                C=1.0,
                kernel=sv_kernel,
                class_weight='balanced',
                gamma='scale',  # 'auto' correct for n_features, scale correct for n_features and X.var
                probability=True,
                tol=1e-3,
                random_state=random_state,
            )
        elif clf.lower() == 'rf' or clf.lower() == 'randomforest':
            clf = RandomForestClassifier(
                n_estimators=1000,  # 500
                max_depth=None,
                min_samples_split=5,
                max_features='sqrt',
                random_state=random_state,
                class_weight='balanced',
            )
        else:
            raise ValueError('Unknown requested string for Clf')
    # set random state    
    np.random.seed(random_state)
    # set cross-validation method with number of folds
    cv = cv_method(n_splits=n_folds, )
    cv.get_n_splits(X_cv, y_cv)

    # define dicts to return results
    y_pred_dict, y_proba_dict, y_true_dict = {}, {}, {}
    og_sample_indices = {}

    if verbose: print(clf)

    # iterate over loops
    for F, (train_index, test_index) in enumerate(
        cv.split(X_cv, y_cv)
    ):

        og_sample_indices[F] = test_index

        # define training and test data per fold
        X_train, X_test = X_cv[train_index], X_cv[test_index]
        y_train, y_test = y_cv[train_index], y_cv[test_index]

        if verbose: print(f'Fold {F}: # of samples: train {len(X_train)}, test {len(X_test)}')
        
        # fit model
        clf.fit(X=X_train, y=y_train)        
        # save predictions for posthoc analysis and conf matrix
        y_proba_dict[F] = clf.predict_proba(X=X_test)
        y_pred_dict[F] = clf.predict(X=X_test)
        y_true_dict[F] = y_test

    return y_pred_dict, y_proba_dict, y_true_dict, og_sample_indices


def multiclass_conf_matrix(
    y_true, y_pred, labels: list = [],
):
    """
    Create Confusion-Matrix for multi-class
    prediction

    Input:
        - y_true: true labels, can be single array
            or list, or dict with values per fold
        - y_pred: predicted labels, can be single array
            or list, or dict with values per fold
        - labels: list with string definitions of
            numerical labels, default: use of numericals
    
    Returns:
        - conf_matr: confusion matrix as pd.crosstab
    """
    # merge cv-folds into single array
    if type(y_true) == dict and type(y_pred) == dict:
        y_pred_all, y_true_all = [], []
        for fold in y_true.keys():
            y_pred_all.extend(y_pred[fold])
            y_true_all.extend(y_true[fold])
        # set back to original variable
        y_true = y_true_all
        y_pred = y_pred_all

    num_labels = np.unique(y_true)

    # if no labels given, use numerical labels
    if len(labels) == 0:
        labels = num_labels.astype(str)
    # transform numerical predictions into labels
    reversefactor = dict(zip(num_labels, labels))
    y_pred = np.vectorize(reversefactor.get)(y_pred)
    y_true = np.vectorize(reversefactor.get)(y_true)

    cm = crosstab(
            y_true, y_pred,
            rownames=['True Scores'],
            colnames=['Predicted Scores'])
    
    # check and correct if all true labels are present in predicted labels
    if cm.shape[0] != cm.shape[1]:
        for i, row_lab in enumerate(cm.index):
            if row_lab not in cm.keys():
                cm.insert(i, row_lab, [0] * cm.shape[0])

    return cm


def get_penalties_from_conf_matr(cm):
    """
    Calculates differences between true and
    predicted labels in confusion matrix

    Input:
        - cm: confusion matrix, can be multiclass

    Returns:
        - mean_pen: mean penalty
        - std_pen: std dev penalty
        - score_penalties: list with all penalties,
            corresponding to n samples
    """
    score_penalties = []
    row_col_combis = list(product(range(len(cm)), range(len(cm))))
    for row, col in row_col_combis:
        penalty = abs(row - col)
        score_penalties.extend([penalty] * cm.values[row, col])

    mean_pen = np.mean(score_penalties)
    std_pen = np.std(score_penalties)
    
    return mean_pen, std_pen, score_penalties