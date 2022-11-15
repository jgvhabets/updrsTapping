"""
ReTap predictive analysis

Prepare features (X) and labels (y) and divide
in training and test sets
"""

# import public pacakges and functions
import numpy as np
import pandas as pd

def select_traces_and_feats(
    ftClass,
    center: str = 'all',
    use_sel_fts=True,
):
    assert center.upper() in ['ALL', 'BER', 'DUS'], (
        'defined center must be "all", "ber", or "dus"'
    )
        
    feats = [
        'nTaps', 'freq',
        'mean_tapRMSnrm', 'coefVar_tapRMSnrm', 'IQR_tapRMSnrm', 'decr_tapRMSnrm', 'slope_tapRMSnrm', 
        'mean_raise_velocity', 'coefVar_raise_velocity', 'IQR_raise_velocity', 'decr_raise_velocity', 'slope_raise_velocity',
        'mean_intraTapInt', 'coefVar_intraTapInt', 'IQR_intraTapInt', 'decr_intraTapInt', 'slope_intraTapInt',
        'mean_jerkiness', 'coefVar_jerkiness', 'IQR_jerkiness', 'decr_jerkiness', 'slope_jerkiness',
        'mean_jerkiness_smooth', 'coefVar_jerkiness_smooth', 'IQR_jerkiness_smooth', 'decr_jerkiness_smooth', 'slope_jerkiness_smooth'
    ]
    if use_sel_fts:
        feats = [
        'nTaps', 'freq',
        'mean_intraTapInt', 'coefVar_intraTapInt',
        'mean_jerkiness', 'IQR_jerkiness', 'coefVar_jerkiness',
        'mean_raise_velocity', 'coefVar_raise_velocity',
        'mean_tapRMSnrm', 'coefVar_tapRMSnrm', 'slope_tapRMSnrm', 
    ]

    zero_taps = [
        'BER023_M1S0_R_3',
        'DUS022_M0S0_L_1',
        'DUS006_M0S0_L_1'
    ]

    avail_traces = ftClass.incl_traces
    avail_traces = [t for t in avail_traces if t not in zero_taps]

    if center.upper() == 'BER' or center.upper() == 'DUS':
        avail_traces = [t for t in avail_traces if center.upper() in t]

    return avail_traces, feats

def create_X_y_vectors(
    ftClass, incl_traces, incl_feats,
    to_norm: bool = False,
):
    # df for feature (X)
    X_df = pd.DataFrame(
        data=np.zeros((
            len(incl_traces),
            len(incl_feats)
        )),
        columns=incl_feats,
    )
    index_order = incl_traces

    # aray for outcome (updrs subscores)
    y = np.array(len(incl_traces) * [np.nan])
    # fill y with values
    for i, trace in enumerate(incl_traces):
        y[i] = getattr(ftClass, trace).tap_score

    # fill X with ft-values    
    for i, trace in enumerate(incl_traces):

        traceClass = getattr(ftClass, trace)
        
        for ft in X_df.keys():
        
            X_df.iloc[i][ft] = getattr(traceClass.fts, ft) 

    # Normalise vector per array-feature over all samples
    if to_norm:
        for ft in X_df.keys():
            vec_max = np.nanmax(X_df[ft])
            X_df[ft] = X_df[ft] / vec_max

    # deal with missings
    # for now set all to zero, ideally: avoid zeros in extraction
    X = X_df.values

    nan_mask = np.isnan(X)
    X[nan_mask] = 0

    assert np.isnan(X).any() == False, print(
        'X array contains missing values:\n',
        np.isnan(X_df).any()
    )

    return X, y