"""
Perform Cross-Validation in prediction model's
development phase.
"""

# import public packages
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC


def get_cvFold_predictions_dicts(
    X_cv,
    y_cv,
    cv_method=StratifiedKFold,
    n_folds: int=4,
    clf=LDA(),
    sv_kernel='linear',
    lr_solver='lbfgs',
    random_state=27,
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
                gamma='scale',  # 'auto' correct for n_features, scale correct for n_features and X.var
                probability=True,
                tol=1e-3,
                random_state=random_state,
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

    print(clf)

    # iterate over loops
    for F, (train_index, test_index) in enumerate(
        cv.split(X_cv, y_cv)
    ):

        og_sample_indices[F] = test_index

        # define training and test data per fold
        X_train, X_test = X_cv[train_index], X_cv[test_index]
        y_train, y_test = y_cv[train_index], y_cv[test_index]

        print(f'# of samples: train {len(X_train)}, test {len(X_test)}')
        print(f'# true labels {sum(y_test)}: chance is {sum(y_test) / len(y_test)}', )
        
        # fit model
        clf.fit(X=X_train, y=y_train)        
        # save predictions for posthoc analysis and conf matrix
        y_proba_dict[F] = clf.predict_proba(X=X_test)
        y_pred_dict[F] = clf.predict(X=X_test)
        y_true_dict[F] = y_test

    return y_pred_dict, y_proba_dict, y_true_dict, og_sample_indices