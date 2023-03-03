"""
ReTap predictive analysis

Helpers functions in classification workflow
"""

# import public pacakges and functions
import numpy as np
import pandas as pd
from itertools import compress

import tap_predict.tap_pred_prepare as pred_prep


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
