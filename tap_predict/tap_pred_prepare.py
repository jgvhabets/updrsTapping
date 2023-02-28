"""
ReTap predictive analysis

Prepare features (X) and labels (y) and divide
in training and test sets
"""

# import public pacakges and functions
import numpy as np
import pandas as pd
from dataclasses import dataclass, field

# own functions
from tap_extract_fts.tapping_postFeatExtr_calc import z_score_array

def select_traces_and_feats(
    ftClass,
    center: str='all',
    use_sel_fts=True,
    excl_traces: list=[],
    excl_subs: list=[],
):
    assert type(excl_subs) == list, 'excl_subs has to list'
    
    assert center.upper() in ['ALL', 'BER', 'DUS'], (
        'defined center must be "all", "ber", or "dus"'
    )
        
    feats = [
        'total_nTaps', 'freq',
        'mean_tapRMSnrm', 'coefVar_tapRMSnrm', 'IQR_tapRMSnrm', 'decr_tapRMSnrm', 'slope_tapRMSnrm', 
        'mean_raise_velocity', 'coefVar_raise_velocity', 'IQR_raise_velocity', 'decr_raise_velocity', 'slope_raise_velocity',
        'mean_intraTapInt', 'coefVar_intraTapInt', 'IQR_intraTapInt', 'decr_intraTapInt', 'slope_intraTapInt',
        'mean_jerkiness', 'coefVar_jerkiness', 'IQR_jerkiness', 'decr_jerkiness', 'slope_jerkiness',
        'mean_jerkiness_smooth', 'coefVar_jerkiness_smooth', 'IQR_jerkiness_smooth', 'decr_jerkiness_smooth', 'slope_jerkiness_smooth'
    ]
    if use_sel_fts:
        feats = [
        'freq', 'mean_intraTapInt', 'coefVar_intraTapInt',
        'mean_jerkiness_taps', 'coefVar_jerkiness_taps', 'jerkiness_trace',
        'mean_raise_velocity', 'coefVar_raise_velocity', 'slope_raise_velocity', 
        'mean_tapRMSnrm', 'coefVar_tapRMSnrm', 'slope_tapRMSnrm',
        'mean_tapRMS', 'coefVar_tapRMS', 'slope_tapRMS', 
    ]

    avail_traces = ftClass.incl_traces
    # filter out traces if defined
    if len(excl_traces) > 0:
        avail_traces = [
            t for t in avail_traces
            if t not in excl_traces
        ]
    
    # filter out traces of subs to excl (if defined)
    if len(excl_subs) > 0:
        for sub_ex in excl_subs:
            avail_traces = [
                t for t in avail_traces
                if sub_ex not in t
            ]

    # select out traces without detected taps
    zero_taps = []
    if len(zero_taps) > 0:
        avail_traces = [t for t in avail_traces if t not in zero_taps]
    # select on center
    if center.upper() == 'BER' or center.upper() == 'DUS':
        avail_traces = [t for t in avail_traces if center.upper() in t]

    return avail_traces, feats


@dataclass(init=True, repr=True,)
class predictionData:
    X: np.ndarray
    y: np.ndarray
    ids: np.ndarray=field(default_factory=np.array([]))


def create_X_y_vectors(
    ftClass,
    incl_feats,
    incl_traces,
    excl_traces = [],
    excl_subs = [],
    to_norm: bool = False,
    to_zscore: bool = False,
    to_mask_4: bool=False,
    to_mask_0: bool=False,
    return_ids: bool=False,
    as_class: bool=False,
    mask_nans=True
):
    """
    create machine learning ready data set, X matrix
    input, y vector labels. Define features to include,
    define to in or exclude specific traces or subjects

    Arguments:
        - ftClass: class with features (FeatureSet)
        - in / excl feats, traces, subs
        - to_norm
        - to_yscore
        - return_ids: return 3rd vector with trace-IDs
            corresponding to X and y
        - mask_nans: mask all NaNs with 0

    Returns:
        - X: input matrix
        - y: vector with true labels
        - ids: vector with trace ids to identify later results
    """
    assert to_norm == False or to_zscore == False, (
        'to_norm AND to_zscore can NOT both be True'
    )
    # filter out traces if defined
    if len(excl_traces) > 0:
        incl_traces = [
            t for t in incl_traces
            if t not in excl_traces
        ]
    
    # filter out traces of subs to excl (if defined)
    if len(excl_subs) > 0:
        for sub_ex in excl_subs:
            incl_traces = [
                t for t in incl_traces
                if sub_ex not in t
            ]

    # fill outcome array y with updrs subscores
    y = [getattr(ftClass, t).tap_score for t in incl_traces]
    y = np.array(y)
    # create corresponding vector with trace ids for later identifying
    ids_vector = np.array(incl_traces)

    # create X matrixc with input features
    X = []
    ft_dict = {}  # fill a preversion dict of X with ft-values

    for ft in incl_feats: ft_dict[ft] = []

    for trace in incl_traces:
        trace_fts = getattr(ftClass, trace).fts

        for ft in incl_feats:
            ft_dict[ft].append(getattr(trace_fts, ft))
    # transform all feats x traces in array with correct shape
    for ft in incl_feats:
        X.append(ft_dict[ft])
    X = np.array(X).T

    assert X.shape[0] == y.shape[0], ('X and y have '
        'different 1st-dimension')

    # Normalise vector per array-feature over all samples
    if to_norm:
        for ft_i in range(X.shape[1]):
            vec_max = np.nanmax(X[:, ft_i])
            X[:, ft_i] = X[:, ft_i] / vec_max
    # Standardise vector per array-feature over all samples
    elif to_zscore:
        for ft_i in range(X.shape[1]):
            X[:, ft_i] = z_score_array(X[:, ft_i])
    

    # deal with missings
    # for now set all to zero, ideally: avoid zeros in extraction
    if mask_nans:
        nan_mask = np.isnan(X)
        print(f'# of NaNs per feat: {sum(nan_mask)}')
        X[nan_mask] = 0
        # check whether masking worked
        assert np.isnan(X).any() == False, print(
            'X array contains missing values:\n',
            np.isnan(X).any()
        )

    if to_mask_4:
        # Mask UPDRS 4 -> 3 merge (too low number)
        mask = y == 4
        y[mask] = 3


    if as_class:
        if return_ids: return predictionData(X=X, y=y, ids=ids_vector)
        else: return predictionData(X=X, y=y)

    else:
        if return_ids: return X, y, ids_vector
        else: return X, y


def split_dataset_on_pred_proba(
    orig_dataset, probas, og_indices, proba_thr
):
    """
    Split data set after prediction has ran,
    based on the generated predicted
    probabilities

    Input:
        - orig_dataset: as predictedData class
        - probas: dict with predicted probabilities per
            sample, based on orig_dataset
        - og_indices: dict with original sample indices
            corresponding to probas
        - proba_thr: threshold for acceptance of probas
    
    Return:
        - data_true: data with probability exceeding
            threshold, as predictedData class
        - data_false: data with probability below
            threshold, as predictedData class
    """
    assert len(og_indices) == len(probas), (
        '# folds of probabilities and indices does not match'
    )
    for fold in og_indices.keys():
        assert len(og_indices[fold]) == len(probas[fold]), (
            '# probabilities and # indices does not match'
        )
    # create array with classification outcome
    clf_decision = np.zeros((orig_dataset.y.shape))

    # loop over single probabilities in all folds
    for fold_n in probas:
        for i_proba, proba in enumerate(probas[fold_n]):
            # set correct index to True (1) if proba > acceptance threshold
            if proba[1] > proba_thr:
                # find corresponding index in original data
                og_idx = og_indices[fold_n][i_proba]
                clf_decision[og_idx] = 1
    
    # select data into true and false
    sel_true = clf_decision.astype(bool)
    data_true = predictionData(
        X=orig_dataset.X[sel_true],
        y=orig_dataset.y[sel_true],
        ids=orig_dataset.ids[sel_true],
    )
    data_false = predictionData(
        X=orig_dataset.X[~sel_true],
        y=orig_dataset.y[~sel_true],
        ids=orig_dataset.ids[~sel_true],
    )

    return data_true, data_false
