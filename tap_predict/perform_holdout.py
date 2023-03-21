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
        assert isinstance(full_modelname, str), ('UNCLUSTERED data and model not present')
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
        print('fast y true', fast_y)
        y_true_dict['slow'] = slow_y
        y_true_dict['fast'] = fast_y
        print('true', y_true_dict)
        print('pred', y_pred_dict)

    return y_pred_dict, y_true_dict


def holdout_reclassification(
    RECLASS_AFTER_RF: str, scores_to_reclass: list,
    recl_label, og_pred_idx, y_pred_dict, y_true_dict,
    RECLASS_FEATS, CLASS_FEATS, X_holdout, ids_holdout, model_name
):
    """
    Performs the reclassification of labels after
    initial RF classifier in hold out setting
    """
    new_y_true_dict, new_y_pred_dict = {}, {}
    new_idx_dict = {}  # dict which original indices belong to which reclass-fold/score
    (new_y_pred_dict['reclass'], new_y_pred_dict['no_reclass'],
     new_y_true_dict['reclass'], new_y_true_dict['no_reclass'],
     new_idx_dict['reclass'], new_idx_dict['no_reclass']
    ) = [], [],[], [], [], []


    # reclass_y_pred = []
    # perform reclassification per predicted outcome group
    feat_sel = [f in RECLASS_FEATS for f in CLASS_FEATS]

    # select correct data to reclass
    for key in y_pred_dict.keys():
        # hold-out predictions saved under key 'holdout'        
        sel_bool = [s in scores_to_reclass for s in y_pred_dict[key]]
        new_y_true_dict['reclass'].extend(np.array(y_true_dict[key])[sel_bool])
        new_idx_dict['reclass'].extend(np.array(og_pred_idx[key])[sel_bool])

        # add not selected samples
        no_reclass_bool = ~np.array(sel_bool)
        # select X and y NOT selected for reclass
        new_y_pred_dict['no_reclass'].extend(np.array(y_pred_dict[key])[no_reclass_bool])
        new_y_true_dict['no_reclass'].extend(np.array(y_true_dict[key])[no_reclass_bool])
        new_idx_dict['no_reclass'].extend(np.array(og_pred_idx[key])[no_reclass_bool])
    
    # select X for reclass based on included ids
    X_bool_sel = [trace_id in new_idx_dict['reclass'] for trace_id in ids_holdout]
    reclass_X = X_holdout[X_bool_sel, :]
    reclass_X = reclass_X[:, feat_sel]  # select out features to include for reclass
       
    # select correct reclass modelname
    reclass_model = (model_name[:-2] +
                     f'_reclass{RECLASS_AFTER_RF.upper()}'
                     f'{recl_label}.P')
    temp_y_pred, _ = perform_holdout(full_X=reclass_X, full_y=new_y_true_dict['reclass'],
                                     full_modelname=reclass_model)
    new_y_pred_dict['reclass'] = temp_y_pred['holdout']
    # # create new y_dicts after all reclass categories
    # y_pred_dict, y_true_dict = {}, {}
    # y_pred_dict['reclass'] = reclass_y_pred

    return new_y_true_dict, new_y_pred_dict, new_idx_dict