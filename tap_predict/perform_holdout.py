"""
Perform HOLDOUT prediction
"""

# import public packages
from os.path import join, exists
from os import makedirs
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from joblib import dump, load

from retap_utils.utils_dataManagement import get_local_proj_dir


def perform_holdout(
    full_X=None, slow_X=None, fast_X=None,
    full_y=None, slow_y=None, fast_y=None,
    full_modelname=None, slow_modelname=None, fast_modelname=None,
):
    """
    Performs HOLDOUT validation

    Input:
        requires either full X, y, and modelname; OR fast + slow
    
    Returns:
        - y_pred_dict: containing holdout key, OR fast and slow keys
        - y_true_dict: containing holdout key, OR fast and slow keys
    """
    # check if either full (unclustered) or clustered data and models are present
    if isinstance(full_X, np.ndarray):
        assert isinstance(full_modelname, str), ('UNCLUISTERED data and model not present')
        split = 'UNCLUSTERED'
    elif isinstance(slow_X, np.ndarray) and isinstance(fast_X, np.ndarray):
        assert np.logical_and(
            isinstance(slow_modelname, str),
            isinstance(fast_modelname, str)
        ), ('fast and slow data and models not present')
        split = 'CLUSTERED'
    else:
        raise ValueError('NO COMPLETE DATA AND MODEL VARIABLES GIVEN')
    
    y_pred_dict, y_true_dict = {}, {}

    if split == 'UNCLUSTERED':
        # predict and add to dict
        clf = load(join(get_local_proj_dir(), 'results', 'models', full_modelname))
        y_pred_dict['holdout'] = clf.predict(full_X)
        # add true labels to dict
        y_true_dict['holdout'] = full_y

    elif split == 'CLUSTERED':
        # predict and add to dict
        slow_clf = load(join(get_local_proj_dir(), 'results', 'models', slow_modelname))
        fast_clf = load(join(get_local_proj_dir(), 'results', 'models', fast_modelname))
        y_pred_dict['slow'] = slow_clf.predict(slow_X)
        y_pred_dict['fast'] = fast_clf.predict(fast_X)
        # add true labels to dict
        y_true_dict['slow'] = slow_y
        y_true_dict['fast'] = fast_y
    
    return y_pred_dict, y_true_dict


