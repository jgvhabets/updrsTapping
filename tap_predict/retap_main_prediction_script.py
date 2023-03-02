"""
ReTap MAIN Classification Script

Perform:
"""


# Importing public packages
from os.path import join
from numpy import array, arange
from itertools import compress
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score as kappa  

# own functions
from tap_extract_fts.main_featExtractionClass import FeatureSet, singleTrace  # mandatory for pickle import
from retap_utils import utils_dataManagement
import retap_utils.get_datasplit as get_split

import tap_predict.tap_pred_prepare as pred_prep
import tap_predict.tap_pred_help as pred_help
import tap_plotting.retap_plot_clusters as plot_cluster

from tap_predict import retap_cv_models as cv_models
from tap_plotting import plot_cv_folds as plot_folds
import tap_plotting.plot_pred_results as plot_results



### SET VARIABLES (load later from json) ###
MAX_TAPS_PER_TRACE = None  # should be None, 10, 15
FT_CLASS_DATE = '20230228'
DATASPLIT = 'CROSSVAL'  # should be CROSSVAL or HOLDOUT
SAVE_TRAINED_MODEL = False
N_RANDOM_SPLIT = 41  # 01.03.23, default None if split has to be found
CLF_CHOICE = 'RF'
TO_PLOT = True
TO_SAVE_FIG = True
FIG_FNAME = 'Clustered_cm_1'

SUBS_EXCL = ['BER028']  # too many missing acc-data
TRACES_EXCL = ['DUS006_M0S0_L_1']  # no score/video

SCORE_FEW_TAPS_3 = True
CUTOFF_TAPS_3 = 9

TO_ZSCORE = True
TO_NORM = False
TO_MASK_4 = True

CLUSTER_ON_FREQ = True
N_CLUSTERS_FREQ = 2

CLASS_FEATS = [
    'trace_RMSn',
    'trace_entropy',
    'jerkiness_trace',
    'coefVar_intraTapInt',
    'slope_intraTapInt',
    'mean_tapRMS',
    'coefVar_tapRMS',
    'mean_impactRMS',
    'coefVar_impactRMS',
    'slope_impactRMS',
    'mean_raise_velocity',
    'coefVar_raise_velocity',
    'coefVar_tap_entropy',
    'slope_tap_entropy',
]

CLUSTER_FEATS = [
    'mean_intraTapInt',
    'coefVar_intraTapInt',
    'freq'
]


#########################
# DATA PREPARATION PART #
#########################


### LOAD FEATURE SET
if MAX_TAPS_PER_TRACE:
    ftClass_name = f'ftClass_max{MAX_TAPS_PER_TRACE}_{FT_CLASS_DATE}.P'
else:
    ftClass_name = f'ftClass_ALL_{FT_CLASS_DATE}.P'  # include all taps per trace

FT_CLASS = utils_dataManagement.load_class_pickle(
    join(utils_dataManagement.get_local_proj_dir(),
         'data', 'derivatives', ftClass_name)
)


### GET DATA SPLIT CROSS-VAL OR HOLD-OUT
# add subjects to exclude to SUBS_EXCL
datasplit_subs = get_split.find_dev_holdout_split(
    feats=FT_CLASS,
    subs_excl=SUBS_EXCL,
    traces_excl=TRACES_EXCL,
    choose_random_split=N_RANDOM_SPLIT
)

if DATASPLIT == 'CROSSVAL':
    datasplit_subs_incl = datasplit_subs['dev']
    datasplit_subs_excl = datasplit_subs['hout']
elif DATASPLIT == 'HOLDOUT':
    datasplit_subs_incl = datasplit_subs['hout']
    datasplit_subs_excl = datasplit_subs['dev']
SUBS_EXCL.extend(datasplit_subs_excl)


### EXCLUDE TRACES with SMALL NUMBER of TAPS,
# and CLASSIFY them AS THREES
if SCORE_FEW_TAPS_3:
    (classf_taps_3,
     y_pred_fewTaps,
     y_true_fewTaps
    ) = pred_help.classify_based_on_nTaps(
        max_n_taps=CUTOFF_TAPS_3,
        ftClass=FT_CLASS,
        in_cv=True)
    # select traces from subs present in current datasplit
    sel = [
        array([t.startswith(s) for s in datasplit_subs_incl]).any()
        for t in classf_taps_3
    ]
    TRACES_EXCL.extend(list(compress(classf_taps_3, sel)))
    y_pred_fewTaps = list(compress(y_pred_fewTaps, sel))
    y_true_fewTaps = list(compress(y_true_fewTaps, sel))

### CREATE DATA FOR PREDICTION (considers datasplit and excl-too-few-taps)
pred_data = pred_prep.create_X_y_vectors(
    FT_CLASS,
    incl_traces=FT_CLASS.incl_traces,
    incl_feats=CLASS_FEATS,
    excl_traces=TRACES_EXCL,
    excl_subs=SUBS_EXCL,  # includes excluding of HOLDOUT OR CROSSVAL
    to_norm=TO_NORM,
    to_zscore=TO_ZSCORE,
    to_mask_4=TO_MASK_4,
    return_ids=True,
    as_class=True,
    mask_nans=True,
)

### SPLIT IN CLUSTERS IF DESIRED
if CLUSTER_ON_FREQ:

    cluster_data = pred_prep.create_X_y_vectors(
        FT_CLASS,
        incl_traces=FT_CLASS.incl_traces,
        incl_feats=CLUSTER_FEATS,
        excl_traces=TRACES_EXCL,
        excl_subs=SUBS_EXCL,  # includes excluding of HOLDOUT OR CROSSVAL
        to_norm=TO_NORM,
        to_zscore=TO_ZSCORE,
        to_mask_4=TO_MASK_4,
        return_ids=True,
        as_class=True,
        mask_nans=True,
    )
    # create cluster labels
    y_clusters, _, _ = plot_cluster.get_kMeans_clusters(
        X=cluster_data.X,
        n_clusters=N_CLUSTERS_FREQ,
        to_zscore=TO_ZSCORE,
        to_norm=TO_NORM,
    )
    # split pred_data in two clusters
    (fast_pred_data, slow_pred_data) = pred_help.split_data_in_clusters(
        pred_data, y_clusters, cluster_data, CLUSTER_FEATS
    )

#######################
# CLASSIFICATION PART #
#######################


### CLASSIFY WITH RANDOM FORESTS

if not CLUSTER_ON_FREQ:
    nFolds = 6

    # generate outcome dict with list per fold
    (y_pred_dict, y_proba_dict,
    y_true_dict, og_pred_idx
    ) = cv_models.get_cvFold_predictions_dicts(
        X_cv=pred_data.X,
        y_cv=pred_data.y,
        cv_method=StratifiedKFold,
        n_folds=nFolds,
        clf=CLF_CHOICE,
        verbose=False,
    )


elif CLUSTER_ON_FREQ:
    nFolds = 3
    y_pred_dict, y_proba_dict, y_true_dict, og_pred_idx = {}, {}, {}, {}

    for c_name, cluster_data in zip(
        ['fast', 'slow'], [fast_pred_data, slow_pred_data]
    ):
        # generate outcome dict with list per fold
        (y_pred_c, _, y_true_c, _
        ) = cv_models.get_cvFold_predictions_dicts(
            X_cv=cluster_data.X,
            y_cv=cluster_data.y,
            cv_method=StratifiedKFold,
            n_folds=nFolds,
            clf=CLF_CHOICE,
            verbose=False,
        )
        # add every fold from cluster to general dict
        for i_d in arange(nFolds):
            y_true_dict[f'{c_name}_{i_d}'] = y_true_c[i_d]
            y_pred_dict[f'{c_name}_{i_d}'] = y_pred_c[i_d]

# add traces classified as 3s as separate dict fold
if SCORE_FEW_TAPS_3:
    y_true_dict['fewtaps'] = y_true_fewTaps
    y_pred_dict['fewtaps'] = y_pred_fewTaps


##################
# SAVING RESULTS #
##################

if TO_MASK_4: mc_labels = ['0', '1', '2', '3-4']
else: mc_labels = ['0', '1', '2', '3', '4']

# create multiclass confusion matrix
cm = cv_models.multiclass_conf_matrix(
    y_true=y_true_dict, y_pred=y_pred_dict,
    labels=mc_labels,
)

# create metrics
y_true_temp, y_pred_temp = [], []

for key in y_true_dict.keys():
    y_true_temp.extend(y_true_dict[key])
    y_pred_temp.extend(y_pred_dict[key])

k_score = kappa(y_true_temp, y_pred_temp, weights='linear')
R, R_p = spearmanr(y_true_temp, y_pred_temp)


plot_results.plot_confMatrix_scatter(
    y_true_temp, y_pred_temp,
    R=R, K=k_score, CM=cm,
    to_save=TO_SAVE_FIG, fname=FIG_FNAME,
)
