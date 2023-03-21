"""
ReTap predictive analysis

Helpers functions in classification workflow
"""

# import public pacakges and functions
import numpy as np
import pandas as pd
from itertools import compress

import tap_predict.tap_pred_prepare as pred_prep

from numpy.random import seed
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

def classify_based_on_nTaps(
    max_n_taps, ftClass, score_to_set=3,
):
    """
    Classify traces based on a too small number of
    taps detected

    Input:
        - max_n_taps: threshold of taps present
        - ftClass: features used
        - score_to_set: score to be classified with
        - in_cv: performed in cross-validation, important
            for true label handling
    
    Returns:
        - selected_traces: names of traces classified, to be
            excluded in further prediction flow
        - pred_scores: predicted scores for seleced traces
        (- true_scores: only given in cross-validation mode)
    """

    cutoff_sel = [len(getattr(ftClass, t).impact_idx) < max_n_taps
                  for t in ftClass.incl_traces]
    
    selected_traces = []
    true_scores = []

    for t in list(compress(ftClass.incl_traces, cutoff_sel)):

        selected_traces.append(t)
        true_scores.append(getattr(ftClass, t).tap_score)

    # masking of true 4 to 3
    true_scores = np.array(true_scores)
    sel = true_scores == 4
    true_scores[sel] = 3

    pred_scores = [score_to_set] * len(selected_traces)

    # print(
    #     f'CLASSIFIED as {score_to_set} based on < {max_n_taps} TAPS:\n\t'
    #     f'{selected_traces}\n\t'
    # )
    # print(f'true scores: {true_scores}')

    return selected_traces, pred_scores, true_scores
    


def define_fast_cluster(y_clust, cluster_fts, cluster_X):
    """
    Returns index of faster cluster
    """
    cluster_mean_ITIs = []
    ft = 'mean_intraTapInt'
    
    for i_cls in np.unique(y_clust):

        i_ft = np.where([f == ft for f in cluster_fts])[0][0]
        mean_iti_cluster = np.mean(cluster_X[y_clust == i_cls, i_ft])
        cluster_mean_ITIs.append(mean_iti_cluster)

        print(f'\tcluster {i_cls}: {mean_iti_cluster}')

    fast_cluster_i = np.argmin(cluster_mean_ITIs)

    return fast_cluster_i


def split_data_in_clusters(
    pred_data, y_clusters, cluster_X, cluster_fts
):
    # Define which cluster contains faster tappers
    fast_cluster_i = define_fast_cluster(
        y_clusters, cluster_fts, cluster_X
    )
    # split in clusters
    if fast_cluster_i == 0: slow_cluster_i = 1
    if fast_cluster_i == 1: slow_cluster_i = 0

    cv_fast_data = pred_prep.predictionData(
        X=pred_data.X[y_clusters == fast_cluster_i],
        y=pred_data.y[y_clusters == fast_cluster_i],
        ids=pred_data.ids[y_clusters == fast_cluster_i])

    cv_slow_data = pred_prep.predictionData(
        X=pred_data.X[y_clusters == slow_cluster_i],
        y=pred_data.y[y_clusters == slow_cluster_i],
        ids=pred_data.ids[y_clusters == slow_cluster_i])

    return cv_fast_data, cv_slow_data


def get_model_param_fig_names(
    CLF_CHOICE, USE_MODEL_DATE, CLUSTER_ON_FREQ,
    MAX_TAPS_PER_TRACE, DATASPLIT, RECLASS_AFTER_RF,
    MASK_0: False, testDev: bool = False,
):
    """
    Create names of Models, pickled feature files,
    model parameters, figures in workflow of
    prediction script retap_main_prediction_script.py 
    
    Input:
        - all variables coming from main_predict script
            all have to be given
    
    Returns:
        - naming_dict: containing all created name-strings
    """
    # dict to save all names in, use dict bcs of variable content
    # depending on input variables
    naming_dict = {}

    if CLUSTER_ON_FREQ:
        naming_dict['FIG_FNAME'] = f'Clustered_{CLF_CHOICE}'
        naming_dict['MODEL_NAME_FAST'] = f'{USE_MODEL_DATE}_{CLF_CHOICE}_CLUSTERED_FAST'
        naming_dict['MODEL_NAME_SLOW'] = f'{USE_MODEL_DATE}_{CLF_CHOICE}_CLUSTERED_SLOW'
        naming_dict['STD_PARAMS'] = f'{USE_MODEL_DATE}_STD_params'
        naming_dict['CLUSTER_STD_PARAMS'] = f'{USE_MODEL_DATE}_STD_params_cluster'

        if MAX_TAPS_PER_TRACE:
            naming_dict['FIG_FNAME'] += f'_{MAX_TAPS_PER_TRACE}taps'
            naming_dict['MODEL_NAME_FAST'] += f'_{MAX_TAPS_PER_TRACE}taps.P'
            naming_dict['MODEL_NAME_SLOW'] += f'_{MAX_TAPS_PER_TRACE}taps.P'
            naming_dict['STD_PARAMS'] += f'_{MAX_TAPS_PER_TRACE}taps.csv'
            naming_dict['CLUSTER_STD_PARAMS'] += f'_{MAX_TAPS_PER_TRACE}taps.csv'
        else:
            naming_dict['FIG_FNAME'] += f'_alltaps'
            naming_dict['MODEL_NAME_FAST'] += f'_alltaps.P'
            naming_dict['MODEL_NAME_SLOW'] += f'_alltaps.P'
            naming_dict['STD_PARAMS'] += f'_alltaps.csv'
            naming_dict['CLUSTER_STD_PARAMS'] += f'_alltaps.csv'

    elif not CLUSTER_ON_FREQ:
        naming_dict['FIG_FNAME'] = f'Unclustered_{CLF_CHOICE}'
        naming_dict['MODEL_NAME'] = f'{USE_MODEL_DATE}_{CLF_CHOICE}_UNCLUSTERED'
        naming_dict['STD_PARAMS'] = f'{USE_MODEL_DATE}_STD_params'

        if MASK_0:
            for key in ['FIG_FNAME', 'MODEL_NAME', 'STD_PARAMS']:
                naming_dict[key] += '_mask0'

        if isinstance(RECLASS_AFTER_RF, str):
            if RECLASS_AFTER_RF.upper() in ['RF', 'LOGREG', 'SVC']:
                naming_dict['FIG_FNAME'] += f'_reclass{RECLASS_AFTER_RF.upper()}'
                # naming_dict['MODEL_NAME'] += f'_reclass{RECLASS_AFTER_RF.upper()}'
            else:
                raise ValueError('incorrect model code for Reclassifying')
            
        if MAX_TAPS_PER_TRACE:
            naming_dict['FIG_FNAME'] += f'_{MAX_TAPS_PER_TRACE}taps'
            naming_dict['MODEL_NAME'] += f'_{MAX_TAPS_PER_TRACE}taps.P'
            naming_dict['STD_PARAMS'] += f'_{MAX_TAPS_PER_TRACE}taps.csv'
        else:
            naming_dict['FIG_FNAME'] += f'_alltaps'
            naming_dict['MODEL_NAME'] += f'_alltaps.P'
            naming_dict['STD_PARAMS'] += f'_alltaps.csv'

    if DATASPLIT == 'HOLDOUT':
        naming_dict['FIG_FNAME'] = 'HOLDOUT_' + naming_dict["FIG_FNAME"]

    if testDev:
        naming_dict['FIG_FNAME'] = 'test_' + naming_dict["FIG_FNAME"]

    return naming_dict


def perform_reclassification(
    RECLASS_AFTER_RF: str, scores_to_reclass: list,
    og_pred_idx, y_pred_dict, y_true_dict,
    RECLASS_FEATS, CLASS_FEATS, pred_data,
):
    """
    Returns:
        - y_true_dict
        - y_pred_dict
        - new_idx_dict: dict with both reclass and no_reclass
            folds, corresponding original indices
        - idx_reclas: only list with indices which are reclassed,
            separate saving for later saving of cv-model
    """
    new_y_true_dict, new_y_pred_dict = {}, {}
    new_idx_dict = {}  # dict which original indices belong to which reclass-fold/score
    (new_y_pred_dict['reclass'], new_y_pred_dict['no_reclass'],
     new_y_true_dict['reclass'], new_y_true_dict['no_reclass'],
     new_idx_dict['reclass'], new_idx_dict['no_reclass']
    ) = [], [],[], [], [], []
    
    # for reclass_i, score_predicted in enumerate([[0, 1], [2,], [3,]]):
        # loop over predicted scores categories
        
    # idx_reclas = []
    # select values to reclassifiy and copy others into new dicts
    for fold in y_pred_dict.keys():
        # print(f'check fold {fold}')
        sel_reclas = [value in scores_to_reclass for value in y_pred_dict[fold]]
        new_idx_dict['reclass'].extend(np.array(og_pred_idx[fold])[sel_reclas])

        not_reclass = ~np.array(sel_reclas)
        new_y_pred_dict['no_reclass'].extend(np.array(y_pred_dict[fold])[not_reclass])
        new_y_true_dict['no_reclass'].extend(np.array(y_true_dict[fold])[not_reclass])
        new_idx_dict['no_reclass'].extend(np.array(og_pred_idx[fold])[not_reclass])

    # reclassed_idx_dict[reclass_i] = idx_reclas
    print(f'total # of {scores_to_reclass} values: {len(new_idx_dict["reclass"])}')
    feat_sel = [f in RECLASS_FEATS for f in CLASS_FEATS]
    reclass_X = pred_data.X[new_idx_dict['reclass'], :]  # select out indices for predicted score
    reclass_X = reclass_X[:, feat_sel]  # select out features to include for reclass
    reclass_y =  pred_data.y[new_idx_dict['reclass']]
    n_folds = len(reclass_y) // 35
    if n_folds == 1 or n_folds == 0: n_folds = 2
    print(f'\treclass X: {reclass_X.shape}, y: {reclass_y.shape}; {n_folds} FOLDS')
    seed(27)  # np.random.seed
    if RECLASS_AFTER_RF.upper() == 'RF':
        clf = RandomForestClassifier(random_state=27, n_estimators=500,
                                        class_weight='balanced',)
    elif RECLASS_AFTER_RF.upper() == 'LOGREG':
        clf = LogisticRegression(random_state=27,solver='lbfgs',)
    elif RECLASS_AFTER_RF.upper() == 'SVC':
        clf = SVC(kernel='linear', class_weight='balanced',
                    gamma='scale', random_state=27,)

    cv = StratifiedKFold(n_splits=n_folds, )
    cv.get_n_splits(reclass_X, reclass_y)
    for F, (train_idx, test_idx) in enumerate(
        cv.split(reclass_X, reclass_y)
    ):
        # define training and test data per fold
        X_train, X_test = reclass_X[train_idx], reclass_X[test_idx]
        y_train, y_test = reclass_y[train_idx], reclass_y[test_idx]
        clf.fit(X=X_train, y=y_train)
        # save predictions for posthoc analysis and conf matrix
        preds = clf.predict(X=X_test)
        # store reclassed scores and corr indices in same order
        new_y_pred_dict['reclass'].extend(preds)
        new_y_true_dict['reclass'].extend(y_test)
        # reclassed_og_idx.extend(np.array(idx_reclas)[test_idx])
        # reclass_cm = cv_models.multiclass_conf_matrix(trues, preds, labels=['0', '1', '2', '3-4'])
        # m, sd, _ = cv_models.get_penalties_from_conf_matr(reclass_cm)
        # print(reclass_cm)
        # print(f'\tpenalties mean: {m}, sd: {sd}')
    
    # # create new dicts for preds and trues
    # new_y_true_dict['reclass'] = reclassed_y_true
    # new_y_pred_dict['reclass'] = reclassed_y_pred

    return new_y_true_dict, new_y_pred_dict, new_idx_dict, new_idx_dict['reclass']