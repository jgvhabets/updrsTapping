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
from sklearn.model_selection import StratifiedKFold
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
from tap_predict.perform_holdout import perform_holdout, holdout_reclassification

import tap_plotting.plot_pred_results as plot_results


### SET VARIABLES (load later from json) ###
# define features to use
FT_CLASS_DATE = '20230228'
MAX_TAPS_PER_TRACE = None  # should be None, 10, 15
# define modeling
DATASPLIT = 'HOLDOUT'  # should be CROSSVAL or HOLDOUT
CLF_CHOICE = 'RF'
CLUSTER_ON_FREQ = False
N_CLUSTERS_FREQ = 2
RECLASS_AFTER_RF = None   # RF, LOGREG, SVC or None
RECLASS_SETTINGS = {'scores': [[1,], [2,]],
                    'labels': ['1', '2']}
N_RANDOM_SPLIT = 41  # 01.03.23, default None if split has to be found
# if holdout, choose model
# USE_MODEL_DATE = '20230303' # DGKN version
USE_MODEL_DATE = '20230321' # changed clustering feats and reclassifying

# define saving or plotting
SAVE_TRAINED_MODEL = True
TO_PLOT = True
TO_SAVE_FIG = True

# std settings and exclusions
SUBS_EXCL = ['BER028']  # too many missing acc-data
TRACES_EXCL = ['DUS006_M0S0_L_1']  # no score/video

SCORE_FEW_TAPS_3 = True
CUTOFF_TAPS_3 = 9

TO_ZSCORE = True
TO_NORM = False
TO_MASK_4 = True
TO_MASK_0 = False

TESTING = True

# build names for models, params and figures to use
naming_dict = pred_help.get_model_param_fig_names(
    CLF_CHOICE=CLF_CHOICE, USE_MODEL_DATE=USE_MODEL_DATE,
    CLUSTER_ON_FREQ=CLUSTER_ON_FREQ, DATASPLIT=DATASPLIT,
    RECLASS_AFTER_RF=RECLASS_AFTER_RF,
    MAX_TAPS_PER_TRACE=MAX_TAPS_PER_TRACE,
    testDev=TESTING, MASK_0=TO_MASK_0,
)

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
    # 'coefVar_intraTapInt',
    # 'freq',
    'mean_tapRMS',
    'trace_RMSn',
]

RECLASS_FEATS = [
    'coefVar_tapRMS',
    'coefVar_impactRMS',
    'coefVar_intraTapInt',
    'slope_intraTapInt',
    'mean_raise_velocity',
    # 'slope_impactRMS',
    'coefVar_tap_entropy',
    'slope_tap_entropy'
]


dd = str(dt.date.today().day).zfill(2)
mm = str(dt.date.today().month).zfill(2)
yyyy = dt.date.today().year
today = f'{yyyy}{mm}{dd}'
naming_dict['FIG_FNAME'] += f'_{today}'


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
                              'results', 'models', naming_dict["STD_PARAMS"]),
                         header=0, index_col=0,)
    if CLUSTER_ON_FREQ:
        cluster_params_df = read_csv(join(utils_dataManagement.get_local_proj_dir(),
                                        'results', 'models',
                                        naming_dict["CLUSTER_STD_PARAMS"]),
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
    to_mask_0=TO_MASK_0,
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
from numpy.random import seed
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

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

        #### RECLASSIFY RANDOM FOREST LABELS WITH DIFFERENT CLASSIFIER (per score)
        """
        reclass predicted scores as [0, 1], [2], [3] separately
        """
        if isinstance(RECLASS_AFTER_RF, str):
            reclassed_idx = {}
            for scores_to_recl, recl_label in zip(
                RECLASS_SETTINGS['scores'], RECLASS_SETTINGS['labels']
            ):
            # reclassify scores in defined list
                (y_true_dict,
                 y_pred_dict,
                 og_pred_idx, 
                 reclassed_idx[recl_label]) = pred_help.perform_reclassification(
                    RECLASS_AFTER_RF=RECLASS_AFTER_RF,
                    scores_to_reclass=scores_to_recl,
                    og_pred_idx=og_pred_idx,
                    y_pred_dict=y_pred_dict, y_true_dict=y_true_dict,
                    RECLASS_FEATS=RECLASS_FEATS,
                    CLASS_FEATS=CLASS_FEATS,
                    pred_data=pred_data,
                )


            # reclassed_y_pred, reclassed_y_true, reclassed_og_idx = [], [], []
            # reclassed_idx_dict = {}  # dict which original indices belong to which reclass-fold/score
            # for reclass_i, score_predicted in enumerate([[0, 1], [2,], [3,]]):
            #     # loop over predicted scores categories
            #     idx_reclas = []
            #     # print('RECLASSIFY PREDICTED SCORES ', score_predicted)
            #     for fold in y_pred_dict.keys():
            #         # print(f'check fold {fold}')
            #         sel_reclas = [value in score_predicted for value in y_pred_dict[fold]]
            #         idx_reclas.extend(array(og_pred_idx[fold])[sel_reclas])

            #     reclassed_idx_dict[reclass_i] = idx_reclas
            #     # print(f'total # of {score_predicted} value: {len(idx_reclas)}')
            #     feat_sel = [f in RECLASS_FEATS for f in CLASS_FEATS]
            #     reclass_X = pred_data.X[idx_reclas, :]  # select out indices for predicted score
            #     reclass_X = reclass_X[:, feat_sel]  # select out features to include for reclass
            #     reclass_y =  pred_data.y[idx_reclas]
            #     n_folds = len(reclass_y) // 35
            #     if n_folds == 1: n_folds = 2
            #     # print(f'\treclass X: {reclass_X.shape}, y: {reclass_y.shape}; {n_folds} FOLDS')
            #     seed(27)  # np.random.seed
            #     if RECLASS_AFTER_RF.upper() == 'RF':
            #         clf = RandomForestClassifier(random_state=27, n_estimators=500,
            #                                      class_weight='balanced',)
            #     elif RECLASS_AFTER_RF.upper() == 'LOGREG':
            #         clf = LogisticRegression(random_state=27,solver='lbfgs',)
            #     elif RECLASS_AFTER_RF.upper() == 'SVC':
            #         clf = SVC(kernel='linear', class_weight='balanced',
            #                   gamma='scale', random_state=27,)

            #     cv = StratifiedKFold(n_splits=n_folds, )
            #     cv.get_n_splits(reclass_X, reclass_y)
            #     for F, (train_idx, test_idx) in enumerate(
            #         cv.split(reclass_X, reclass_y)
            #     ):
            #         # define training and test data per fold
            #         X_train, X_test = reclass_X[train_idx], reclass_X[test_idx]
            #         y_train, y_test = reclass_y[train_idx], reclass_y[test_idx]
            #         clf.fit(X=X_train, y=y_train)
            #         # save predictions for posthoc analysis and conf matrix
            #         preds = clf.predict(X=X_test)
            #         # store reclassed scores and corr indices in same order
            #         reclassed_y_pred.extend(preds)
            #         reclassed_y_true.extend(y_test)
            #         reclassed_og_idx.extend(array(idx_reclas)[test_idx])
            #         # reclass_cm = cv_models.multiclass_conf_matrix(trues, preds, labels=['0', '1', '2', '3-4'])
            #         # m, sd, _ = cv_models.get_penalties_from_conf_matr(reclass_cm)
            #         # print(reclass_cm)
            #         # print(f'\tpenalties mean: {m}, sd: {sd}')
            # # create new dicts for preds and trues
            # y_true_dict, y_pred_dict = {}, {}
            # y_true_dict['reclass'] = reclassed_y_true
            # y_pred_dict['reclass'] = reclassed_y_pred



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


elif DATASPLIT == 'HOLDOUT':
    
    if not CLUSTER_ON_FREQ:
        y_pred_dict, y_true_dict = perform_holdout(
            full_X=pred_data.X, full_y=pred_data.y,
            full_modelname=naming_dict['MODEL_NAME']
        )

        if isinstance(RECLASS_AFTER_RF, str):
            # reclass_y_pred, reclass_y_true = [], []
            # reclassed_idx = {}
            og_pred_idx = {}
            og_pred_idx['holdout'] = pred_data.ids  # add for reclassification

            # perform reclassification per predicted outcome group
            feat_sel = [f in RECLASS_FEATS for f in CLASS_FEATS]

            # for reclass_i, og_preds in enumerate([[0, 1], [2,], [3,]]):
            for scores_to_recl, recl_label in zip(
                RECLASS_SETTINGS['scores'], RECLASS_SETTINGS['labels']
            ):
                (y_true_dict,
                 y_pred_dict,
                 og_pred_idx) = holdout_reclassification(
                    RECLASS_AFTER_RF=RECLASS_AFTER_RF,
                    scores_to_reclass=scores_to_recl,
                    recl_label=recl_label,
                    X_holdout=pred_data.X,
                    ids_holdout=pred_data.ids,
                    og_pred_idx=og_pred_idx,
                    y_pred_dict=y_pred_dict,
                    y_true_dict=y_true_dict,
                    RECLASS_FEATS=RECLASS_FEATS,
                    CLASS_FEATS=CLASS_FEATS,
                    model_name=naming_dict['MODEL_NAME']
                )


            #     # select correct data to reclass
            #     sel_idx = [s in scores_to_recl for s in y_pred_dict['holdout']]
            #     reclass_X = pred_data.X[sel_idx, :]  # select out indices for predicted score
            #     reclass_X = reclass_X[:, feat_sel]  # select out features to include for reclass
            #     reclass_y =  pred_data.y[sel_idx]
            #     # select correct reclass modelname
            #     reclass_model = (naming_dict['MODEL_NAME'][:-2] +
            #                      f'_reclass{RECLASS_AFTER_RF.upper()}'
            #                      f'{recl_label}.P')
            #     temp_y_pred, _ = perform_holdout(full_X=reclass_X, full_y=reclass_y,
            #                                       full_modelname=reclass_model)
            #     reclass_y_pred.extend(temp_y_pred['holdout'])
            #     reclass_y_true.extend(reclass_y)
            # # create new y_dicts after all reclass categories
            # y_pred_dict, y_true_dict = {}, {}
            # y_pred_dict['reclass'] = reclass_y_pred
            # y_true_dict['reclass'] = reclass_y_true

    
    elif CLUSTER_ON_FREQ:
        y_pred_dict, y_true_dict = perform_holdout(
            slow_X=slow_pred_data.X, slow_y=slow_pred_data.y,
            fast_X=fast_pred_data.X, fast_y=fast_pred_data.y,
            slow_modelname=naming_dict['MODEL_NAME_SLOW'],
            fast_modelname=naming_dict['MODEL_NAME_FAST'],
        )

# add traces classified on few-taps as separate dict fold
if SCORE_FEW_TAPS_3:
    y_true_dict['fewtaps'] = y_true_fewTaps
    y_pred_dict['fewtaps'] = y_pred_fewTaps


###################################
# SAVE CROSSVAL MODEL FOR HOLDOUT #
###################################

if SAVE_TRAINED_MODEL and DATASPLIT == 'CROSSVAL':
    
    # save std parameters for classification
    fname = f'{today}_STD_params'
    if TO_MASK_0: fname += '_mask0' 
    if MAX_TAPS_PER_TRACE: fname += f'_{MAX_TAPS_PER_TRACE}taps'
    else: fname += '_alltaps'
    STD_params.to_csv(
        join(utils_dataManagement.get_local_proj_dir(), 'results', 'models',
             f'{fname}.csv'),
        header=True, index=True
    )
    # save std parameters for clustering
    if CLUSTER_ON_FREQ:
        fname = f'{today}_STD_params_cluster'
        if MAX_TAPS_PER_TRACE: fname += f'_{MAX_TAPS_PER_TRACE}taps'
        else: fname += '_alltaps'
        STD_params_cluster.to_csv(
            join(utils_dataManagement.get_local_proj_dir(), 'results', 'models',
                 f'{fname}.csv'),
            header=True, index=True
        )

    # save model trained on FULL crossvalidation data
    elif not CLUSTER_ON_FREQ:
        # save general Random Forest model
        model_fname = f'{today}_{CLF_CHOICE}_UNCLUSTERED'
        if TO_MASK_0: model_fname += '_mask0' 
        if MAX_TAPS_PER_TRACE: model_fname += f'_{MAX_TAPS_PER_TRACE}taps'
        else: model_fname += f'_alltaps'

        saveload_models.save_model_in_cv(
            clf=CLF_CHOICE, X_CV=pred_data.X, y_CV=pred_data.y,
            # path=utils_dataManagement.find_onedrive_path('models'),  # saves default to local project folder
            model_fname=model_fname,
        )

        # save reclassing models per fold / score-category [0: 0-1, 1: 2, 2: 3]
        if isinstance(RECLASS_AFTER_RF, str):
            feat_sel = [f in RECLASS_FEATS for f in CLASS_FEATS]

            for score_to_reclas in reclassed_idx.keys():
                idx_reclas = reclassed_idx[score_to_reclas]
                reclass_modelname = (model_fname +
                                     f'_reclass{RECLASS_AFTER_RF.upper()}'
                                     f'{score_to_reclas}')
                reclass_X = pred_data.X[idx_reclas, :]  # select out indices for predicted score
                reclass_X = reclass_X[:, feat_sel]  # select out features to include for reclass
                reclass_y =  pred_data.y[idx_reclas]
                saveload_models.save_model_in_cv(
                    clf=RECLASS_AFTER_RF,
                    X_CV=reclass_X, y_CV=reclass_y,
                    # path=utils_dataManagement.find_onedrive_path('models'),  # saves default to local project folder
                    model_fname=reclass_modelname,
                )

    elif CLUSTER_ON_FREQ:
        for c_name, cluster_data in zip(
            ['fast', 'slow'], [fast_pred_data, slow_pred_data]
        ):
            
            model_fname = f'{today}_{CLF_CHOICE}_CLUSTERED_{c_name.upper()}'
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

if TO_MASK_4:
    if TO_MASK_0: mc_labels = ['0-1', '1', '2', '3-4']
    else: mc_labels = ['0', '1', '2', '3-4']
else: mc_labels = ['0', '1', '2', '3', '4']

# save slow and fast clusters seperately
if CLUSTER_ON_FREQ:
    for key in y_pred_dict:
        if key not in ['slow', 'fast']: continue
        k_score = kappa(y_true_dict[key], y_pred_dict[key], weights='linear')
        R, R_p = spearmanr(y_true_dict[key], y_pred_dict[key])
        cmtemp = cv_models.multiclass_conf_matrix(
            y_true=y_true_dict[key], y_pred=y_pred_dict[key],
            labels=mc_labels,
        )
        print(key, cmtemp)
        plot_results.plot_confMatrix_scatter(
            y_true_dict[key], y_pred_dict[key],
            R=R, K=k_score, CM=cmtemp,
            to_save=TO_SAVE_FIG,
            fname=f'{naming_dict["FIG_FNAME"]}_{key.upper()}',
        )

# create multiclass confusion matrix for all cross-val folds
cm = cv_models.multiclass_conf_matrix(
    y_true=y_true_dict, y_pred=y_pred_dict,
    labels=mc_labels,
)

# create metrics for whole data split

from sklearn.metrics import cohen_kappa_score as kappa
from scipy.stats import spearmanr, pearsonr
from pingouin import intraclass_corr

y_true_all, y_pred_all = [], []

for key in y_true_dict.keys():
    y_true_all.extend(y_true_dict[key])
    y_pred_all.extend(y_pred_dict[key])

k_score = kappa(y_true_all, y_pred_all, weights='linear')
R, R_p = spearmanr(y_true_all, y_pred_all)
pearsonR, prsR_p = pearsonr(y_true_all, y_pred_all)


# calculate ICC
icc_scores = y_true_all + y_pred_all
icc_judges = ['clin'] * len(y_true_all) + ['model'] * len(y_pred_all)
icc_ids = [str(i) for i in range(len(y_true_all))] * 2
try:
    icc_dat = DataFrame(array([icc_ids, icc_judges, icc_scores]),
                    columns=['IDs', 'Judges', 'Scores'])
except:
    icc_dat = DataFrame(array([icc_ids, icc_judges, icc_scores]).T,
                    columns=['IDs', 'Judges', 'Scores'])
    print('transposed ICC dat')

icc = intraclass_corr(data=icc_dat, targets='IDs', raters='Judges',
                         ratings='Scores')
icc_score = icc.iloc[5]['ICC']  # 5 row is two-way, mixed effect, k-raters
print(icc)

print(f'Kappa {k_score}, Spearman R: {R}, p={R_p}, Pearson R: {pearsonR}, p={prsR_p}')

if TO_PLOT:
    plot_results.plot_confMatrix_scatter(
        y_true=y_true_all, y_pred=y_pred_all, plot_box=True,
        R=pearsonR, K=k_score, CM=cm, icc=icc_score, R_meth='Pearson',
        to_save=TO_SAVE_FIG, fname=naming_dict["FIG_FNAME"],
    )
