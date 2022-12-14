"""
ReTap predictive analysis

Prepare features (X) and labels (y) and divide
in training and test sets
"""

# import public pacakges and functions
import numpy as np
import pandas as pd

# own functions
from tap_extract_fts.tapping_postFeatExtr_calc import z_score_array

def select_traces_and_feats(
    ftClass,
    center: str = 'all',
    use_sel_fts=True,
    excl_traces: list = [],
):
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

    # select out traces without detected taps
    zero_taps = []
    if len(zero_taps) > 0:
        avail_traces = [t for t in avail_traces if t not in zero_taps]
    # select on center
    if center.upper() == 'BER' or center.upper() == 'DUS':
        avail_traces = [t for t in avail_traces if center.upper() in t]

    return avail_traces, feats



def create_X_y_vectors(
    ftClass,
    incl_feats,
    incl_traces,
    excl_traces: list = [],
    to_norm: bool = False,
    to_zscore: bool = False,
):
    """
    
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

    # fill outcome array y with updrs subscores
    y = [getattr(ftClass, t).tap_score for t in incl_traces]
    # y = np.array([y]).T
    y = np.array(y)

    # create X matrixc with input features
    X = []
    ft_dict = {}  # fill a preversion dict of X with ft-values
    for ft in incl_feats: ft_dict[ft] = []

    for trace in incl_traces:
        trace_fts = getattr(ftClass, trace).fts

        for ft in incl_feats:
            ft_dict[ft].append(getattr(trace_fts, ft))
            # except KeyError: print(trace, ft)
            
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
    nan_mask = np.isnan(X)
    print(f'# of NaNs per feat: {sum(nan_mask)}')
    X[nan_mask] = 0

    assert np.isnan(X).any() == False, print(
        'X array contains missing values:\n',
        np.isnan(X).any()
    )

    return X, y