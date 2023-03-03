"""
ReTap MAIN Classification Script

cmnd line, from repo path (WIN):
    python -m tap_predict.retap_main_prediction_script
"""


# Importing public packages
from os.path import join
import pickle
from numpy import array, arange
from pandas import DataFrame, read_csv
from itertools import compress
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score as kappa
import datetime as dt

# own functions
from tap_extract_fts.main_featExtractionClass import FeatureSet, singleTrace  # mandatory for pickle import
from retap_utils import utils_dataManagement
import retap_utils.get_datasplit as get_split

import tap_predict.tap_pred_prepare as pred_prep
import tap_predict.tap_pred_help as pred_help
import tap_plotting.retap_plot_clusters as plot_cluster

from tap_predict import retap_cv_models as cv_models
from tap_predict import save_load_pred_models as saveload_models
from tap_predict.perform_holdout import perform_holdout

import tap_plotting.plot_pred_results as plot_results


### SET VARIABLES (load later from json) ###
FT_CLASS_DATE = '20230228'
DATASPLIT = 'HOLDOUT'  # should be CROSSVAL or HOLDOUT
CLUSTER_ON_FREQ = True
N_CLUSTERS_FREQ = 2
MAX_TAPS_PER_TRACE = None  # should be None, 10, 15
CLF_CHOICE = 'RF'
SAVE_TRAINED_MODEL = True
N_RANDOM_SPLIT = 41  # 01.03.23, default None if split has to be found
TO_PLOT = True
TO_SAVE_FIG = True

SUBS_EXCL = ['BER028']  # too many missing acc-data
TRACES_EXCL = ['DUS006_M0S0_L_1']  # no score/video

USE_MODEL_DATE = '20230303' 

SCORE_FEW_TAPS_3 = True
CUTOFF_TAPS_3 = 9

TO_ZSCORE = True
TO_NORM = False
TO_MASK_4 = True


# build names for models, params and figures to use
if CLUSTER_ON_FREQ:
    FIG_FNAME = f'Clustered_{CLF_CHOICE}'
    MODEL_NAME_FAST = f'{USE_MODEL_DATE}_{CLF_CHOICE}_CLUSTERED_FAST'
    MODEL_NAME_SLOW = f'{USE_MODEL_DATE}_{CLF_CHOICE}_CLUSTERED_SLOW'
    STD_PARAMS = f'{USE_MODEL_DATE}_STD_params'
    CLUSTER_STD_PARAMS = f'{USE_MODEL_DATE}_STD_params_cluster'

    if MAX_TAPS_PER_TRACE:
        FIG_FNAME += f'_{MAX_TAPS_PER_TRACE}taps'
        MODEL_NAME_FAST = f'{MODEL_NAME_FAST}_{MAX_TAPS_PER_TRACE}taps.P'
        MODEL_NAME_SLOW = f'{MODEL_NAME_SLOW}_{MAX_TAPS_PER_TRACE}taps.P'
        STD_PARAMS = f'{STD_PARAMS}_{MAX_TAPS_PER_TRACE}taps.csv'
        CLUSTER_STD_PARAMS = f'{CLUSTER_STD_PARAMS}_{MAX_TAPS_PER_TRACE}taps.csv'
    else:
        FIG_FNAME += f'_alltaps'
        MODEL_NAME_FAST = f'{MODEL_NAME_FAST}_alltaps.P'
        MODEL_NAME_SLOW = f'{MODEL_NAME_SLOW}_alltaps.P'
        STD_PARAMS = f'{STD_PARAMS}_alltaps.csv'
        CLUSTER_STD_PARAMS = f'{CLUSTER_STD_PARAMS}_alltaps.csv'

if not CLUSTER_ON_FREQ:
    FIG_FNAME = f'Unclustered_{CLF_CHOICE}'
    MODEL_NAME = f'{USE_MODEL_DATE}_{CLF_CHOICE}_UNCLUSTERED'
    STD_PARAMS = f'{USE_MODEL_DATE}_STD_params'

    if MAX_TAPS_PER_TRACE:
        FIG_FNAME += f'_{MAX_TAPS_PER_TRACE}taps'
        MODEL_NAME = f'{MODEL_NAME}_{MAX_TAPS_PER_TRACE}taps.P'
        STD_PARAMS = f'{STD_PARAMS}_{MAX_TAPS_PER_TRACE}taps.csv'
    else:
        FIG_FNAME += f'_alltaps'
        MODEL_NAME = f'{MODEL_NAME}_alltaps.P'
        STD_PARAMS = f'{STD_PARAMS}_alltaps.csv'

if DATASPLIT == 'HOLDOUT': FIG_FNAME = f'HOLDOUT_{FIG_FNAME}'

testDev = False
if testDev: FIG_FNAME = f'test_{FIG_FNAME}'


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
datasplit_subs = get_split.find_dev_holdout_split(
    feats=FT_CLASS,
    subs_excl=SUBS_EXCL,
    traces_excl=TRACES_EXCL,
    choose_random_split=N_RANDOM_SPLIT
)

if DATASPLIT == 'CROSSVAL':
    datasplit_subs_incl = datasplit_subs['dev']
    datasplit_subs_excl = datasplit_subs['hout']
    params_df=None  # necessary for function to run in cv
    cluster_params_df=None  # necessary for function to run in cv
elif DATASPLIT == 'HOLDOUT':
    datasplit_subs_incl = datasplit_subs['hout']
    datasplit_subs_excl = datasplit_subs['dev']
    params_df = read_csv(join(utils_dataManagement.get_local_proj_dir(),
                              'results', 'models', STD_PARAMS),
                         header=0, index_col=0,)
    if CLUSTER_ON_FREQ:
        cluster_params_df = read_csv(join(utils_dataManagement.get_local_proj_dir(),
                                        'results', 'models', CLUSTER_STD_PARAMS),
                                    header=0, index_col=0,)
else:
    raise ValueError('DATASPLIT has to be CROSSVAL or HOLDOUT')

# add subjects from other data-split to SUBS_EXCL
SUBS_EXCL.extend(datasplit_subs_excl)


### EXCLUDE TRACES with SMALL NUMBER of TAPS, CLASSIFY them AS "3"
if SCORE_FEW_TAPS_3:
    (classf_taps_3,
     y_pred_fewTaps,
     y_true_fewTaps
    ) = pred_help.classify_based_on_nTaps(
        max_n_taps=CUTOFF_TAPS_3,
        ftClass=FT_CLASS,
        )
    # select traces from subs present in current datasplit
    sel = [
        array([t.startswith(s) for s in datasplit_subs_incl]).any()
        for t in classf_taps_3
    ]
    TRACES_EXCL.extend(list(compress(classf_taps_3, sel)))  # exclude classified traces from prediction
    y_pred_fewTaps = list(compress(y_pred_fewTaps, sel))  # store predicted value and true values
    y_true_fewTaps = list(compress(y_true_fewTaps, sel))

### CREATE DATA FOR PREDICTION (considers datasplit and excl-too-few-taps)
pred_data = pred_prep.create_X_y_vectors(
    FT_CLASS,
    incl_traces=FT_CLASS.incl_traces,
    incl_feats=CLASS_FEATS,
    excl_traces=TRACES_EXCL,
    excl_subs=SUBS_EXCL,  # contains data-split exclusion and too-few-tap-exclusion
    to_norm=TO_NORM,
    to_zscore=TO_ZSCORE,
    to_mask_4=TO_MASK_4,
    return_ids=True,
    as_class=True,
    mask_nans=True,
    save_STD_params=SAVE_TRAINED_MODEL,
    use_STD_params_df=params_df,  # only gives ft-mean/-sd in HOLDOUT
)

# split data and params from tuple 
if SAVE_TRAINED_MODEL:
    STD_params = DataFrame(data=pred_data[1], index=CLASS_FEATS,
                           columns=['mean', 'std'])
    pred_data = pred_data[0]

### SPLIT IN CLUSTERS IF DESIRED
if CLUSTER_ON_FREQ:

    cluster_data = pred_prep.create_X_y_vectors(
        FT_CLASS,
        incl_traces=FT_CLASS.incl_traces,
        incl_feats=CLUSTER_FEATS,
        excl_traces=TRACES_EXCL,
        excl_subs=SUBS_EXCL,  # includes excluding of HOLDOUT OR CROSSVAL
        to_zscore=TO_ZSCORE,
        to_mask_4=TO_MASK_4,
        return_ids=True,
        as_class=True,
        save_STD_params=SAVE_TRAINED_MODEL,
        use_STD_params_df=cluster_params_df,  # only gives ft-mean/-sd in HOLDOUT
    )
    # split data and params from tuple 
    if SAVE_TRAINED_MODEL:
        STD_params_cluster = DataFrame(
            data=cluster_data[1], index=CLUSTER_FEATS, columns=['mean', 'std']
        )
        cluster_data = cluster_data[0]
    
    # create cluster labels
    y_clusters, _, _ = plot_cluster.get_kMeans_clusters(
        X=cluster_data.X,
        n_clusters=N_CLUSTERS_FREQ,
    )
    # split pred_data in two clusters
    (fast_pred_data, slow_pred_data) = pred_help.split_data_in_clusters(
        pred_data, y_clusters, cluster_data.X, CLUSTER_FEATS
    )

#######################
# CLASSIFICATION PART #
#######################


### CLASSIFY WITH RANDOM FORESTS

if DATASPLIT == 'CROSSVAL':
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

if DATASPLIT == 'HOLDOUT':
    
    if not CLUSTER_ON_FREQ:
        y_true_dict, y_pred_dict = perform_holdout(
            full_X=pred_data.X, full_y=pred_data.y,
            full_modelname=MODEL_NAME
        )
    
    elif CLUSTER_ON_FREQ:
        y_true_dict, y_pred_dict = perform_holdout(
            slow_X=slow_pred_data.X, slow_y=slow_pred_data.y,
            fast_X=fast_pred_data.X, fast_y=fast_pred_data.y,
            slow_modelname=MODEL_NAME_SLOW,
            fast_modelname=MODEL_NAME_FAST,
        )

# add traces classified on few-taps as separate dict fold
if SCORE_FEW_TAPS_3:
    y_true_dict['fewtaps'] = y_true_fewTaps
    y_pred_dict['fewtaps'] = y_pred_fewTaps


###################################
# SAVE CROSSVAL MODEL FOR HOLDOUT #
###################################

if SAVE_TRAINED_MODEL and DATASPLIT == 'CROSSVAL':
    dd = str(dt.date.today().day).zfill(2)
    mm = str(dt.date.today().month).zfill(2)
    yyyy = dt.date.today().year
    
    # save std parameters for classification
    fname = f'{yyyy}{mm}{dd}_STD_params'
    if MAX_TAPS_PER_TRACE: fname += f'_{MAX_TAPS_PER_TRACE}taps'
    else: fname += '_alltaps'
    STD_params.to_csv(
        join(utils_dataManagement.get_local_proj_dir(), 'results', 'models',
             f'{fname}.csv'),
        header=True, index=True
    )
    # save std parameters for clustering
    if CLUSTER_ON_FREQ:
        fname = f'{yyyy}{mm}{dd}_STD_params_cluster'
        if MAX_TAPS_PER_TRACE: fname += f'_{MAX_TAPS_PER_TRACE}taps'
        else: fname += '_alltaps'
        STD_params_cluster.to_csv(
            join(utils_dataManagement.get_local_proj_dir(), 'results', 'models',
                 f'{fname}.csv'),
            header=True, index=True
        )

    # save model trained on FULL crossvalidation data
    if not CLUSTER_ON_FREQ:
        model_fname = f'{yyyy}{mm}{dd}_{CLF_CHOICE}_UNCLUSTERED'
        if MAX_TAPS_PER_TRACE: model_fname += f'_{MAX_TAPS_PER_TRACE}taps'
        else: model_fname += f'_alltaps'

        saveload_models.save_model_in_cv(
            clf=CLF_CHOICE, X_CV=pred_data.X, y_CV=pred_data.y,
            # path=utils_dataManagement.find_onedrive_path('models'),  # saves default to local project folder
            model_fname=model_fname,
        )
    
    elif CLUSTER_ON_FREQ:
        for c_name, cluster_data in zip(
            ['fast', 'slow'], [fast_pred_data, slow_pred_data]
        ):
            
            model_fname = f'{yyyy}{mm}{dd}_{CLF_CHOICE}_CLUSTERED_{c_name.upper()}'
            if MAX_TAPS_PER_TRACE: model_fname += f'_{MAX_TAPS_PER_TRACE}taps'
            else: model_fname += f'_alltaps'
        
            saveload_models.save_model_in_cv(
                clf=CLF_CHOICE, X_CV=cluster_data.X, y_CV=cluster_data.y,
                # path=utils_dataManagement.find_onedrive_path('models'),  # saves default to local project folder
                model_fname=model_fname,
            )

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

if TO_SAVE_FIG:
    dd = str(dt.date.today().day).zfill(2)
    mm = str(dt.date.today().month).zfill(2)
    yyyy = dt.date.today().year
    FIG_FNAME += f'_{yyyy}{mm}{dd}'


plot_results.plot_confMatrix_scatter(
    y_true_temp, y_pred_temp,
    R=R, K=k_score, CM=cm,
    to_save=TO_SAVE_FIG, fname=FIG_FNAME,
)
