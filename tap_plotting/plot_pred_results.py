


# import function
import os
import numpy as np
from pandas import Series
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score as kappa  
from scipy.stats import spearmanr

from tap_predict import retap_cv_models as cv_models
from retap_utils.utils_dataManagement import find_onedrive_path




def plot_confMatrix_scatter(
    y_true, y_pred, trace_ids,
    R=None, K=None, CM=None, icc=None, R_meth=None,
    plot_scatter=True, plot_box=False, plot_violin=False,
    mc_labels=['0', '1', '2', '3'], og_pred_idx=None,
    subs_in_holdout=None, add_folder=None,
    to_save=False, fname=None, to_show=False,
    datasplit=None,
):
    mean_pen, std_pen, _ = cv_models.get_penalties_from_conf_matr(CM)

    if datasplit == 'HOLDOUT':
        n_plots = 3
        gridspecs = {'width_ratios': [3, 3, 1]}
    else: 
        n_plots = 2
        gridspecs = {'width_ratios': [3, 3]}
    fig, axes = plt.subplots(1, n_plots, figsize=(16, 6),
                            gridspec_kw=gridspecs,)
    fs=20

    if plot_scatter:
        jitt = np.random.uniform(low=-.2, high=0.2, size=len(y_true))
        jitt2 = np.random.uniform(low=-.2, high=0.2, size=len(y_true))
        axes[0].scatter(y_pred+jitt2, y_true+jitt,
                        alpha=.2, s=100,)
    
    if plot_box or plot_violin:
        
        box_lists = [[], [], [], []]  # for boxplots, fill 4 lists with true scores per predicted score in lists
        for pred_value, true_value in zip(y_pred, y_true):
            try:
                box_lists[pred_value].append(true_value)  # append true-value (Y-axis) to the list of the predicted value (X-axis)
            except IndexError:
                print(pred_value, true_value)
                raise ValueError('STOPPPPPPP')
        if plot_box:
            medianprops = dict(linestyle=None, linewidth=0,)
            meanlineprops = dict(linestyle='-', linewidth=2.5,
                                 color='firebrick')
            axes[0].boxplot(box_lists,
                            positions=range(len(mc_labels)),
                            meanline=True, showmeans=True,
                            whis=.5, medianprops=medianprops,
                            meanprops=meanlineprops,)

        if plot_violin:
            viol_parts = axes[0].violinplot(box_lists, positions=range(4),
                            showmeans=True, showextrema=False,
                                showmedians=False,)
            viol_parts['cmeans'].set_facecolor('firebrick')
            viol_parts['cmeans'].set_linewidth(3)
            viol_parts['cmeans'].set_alpha(.8)

    axes[0].set_ylabel('True Tap Score', weight='bold', fontsize=fs)
    axes[0].set_xlabel('Predicted Tap Score', weight='bold', fontsize=fs)
    axes[0].set_xticks(range(4))
    axes[0].set_xticklabels(mc_labels, weight='bold', fontsize=fs)
    axes[0].set_yticks(range(4))
    axes[0].set_yticklabels(mc_labels, weight='bold', fontsize=fs)
    axes[0].spines[['right', 'top']].set_visible(False)


    ### MAKE HEATMAP of conf matrix

    # add missing data in CM table
    for i_label, l in enumerate(mc_labels):
        if l not in CM.index:
            CM.loc[l] = [0] * CM.shape[1]
        if l not in CM.keys():
            CM.insert(loc=i_label, value=[0] * CM.shape[0], column=l)
    CM = CM.sort_index()

    im = axes[1].imshow(CM.values)

    # Show all ticks and label them with the respective list entries
    # axes[1].xaxis.tick_top()
    axes[1].set_xticks(np.arange(len(mc_labels)))
    axes[1].set_yticks(np.arange(len(mc_labels)))
    axes[1].set_xticklabels(mc_labels, weight='bold', fontsize=fs)
    axes[1].set_yticklabels(mc_labels, weight='bold', fontsize=fs, )
    # axes[1].xaxis.set_label_position('top')
    axes[1].set_xlabel('Predicted Tap Score', weight='bold', fontsize=fs, )
    axes[1].set_ylabel('True Tap Score', weight='bold', fontsize=fs)

    # Loop over data dimensions and create text annotations.
    for i in range(len(mc_labels)):
        for j in range(len(mc_labels)):
            value = CM.values[i, j]
            if value > 30: txt_col ='k'
            # elif value > 15 and np.max(CM.values) < 50: txt_col = 'gray'
            else: txt_col = 'w'
            text = axes[1].text(
                j, i, value, weight='bold', fontsize=fs,
                ha="center", va="center", color=txt_col,
      )

    # ANNOTATE METRICS  --> print pred results below conf matrix
    # axes[1].annotate(
    #     (f'Mean Prediction Error: {mean_pen.round(2)} (sd: {std_pen.round(2)})'
    #     f'\n{R_meth} R: {R.round(2)},\nICC: {icc.round(2)}, Kappa: {K.round(2)}'),
    #     xy=(.5, -.3), ha='center', va='top',
    #     xycoords='axes fraction',
    #     fontsize=fs - 2,)
    
    #### PLOT AXIS WITH INDIV R's
    if datasplit == 'HOLDOUT':
        axes[2] = plot_indiv_Rs_holdout(
            ax=axes[2], y_true_list=y_true, y_pred_list=y_pred,
            trace_ids_list=trace_ids, subs_in_holdout=subs_in_holdout,
            fs=fs,
        )

    plt.tight_layout(w_pad=.7, pad=.7)

    if to_save:
        path = os.path.join(
            find_onedrive_path('figures'),
            'prediction'
        )
        if add_folder: path = os.path.join(path, add_folder)
        plt.savefig(os.path.join(path, fname), dpi=300, facecolor='w',)
        print(f'Saved Fig "{fname}" in {path}')
    plt.close()

from scipy.stats import pearsonr, spearmanr


def plot_indiv_Rs_holdout(
    ax, y_true_list, y_pred_list, trace_ids_list,
    subs_in_holdout, fs=14, r_method=pearsonr,
    BER_clr='purple', DUS_clr='darkgreen',
):
    (sub_Rs,
     sub_samples,
     sub_ids,
     _,
     _) = get_indiv_holdout_results(
        y_true_list, y_pred_list, trace_ids_list,
        subs_in_holdout, r_method=r_method,
    )
    #### plot individual R's
    ax.axhline(0, xmin=0, xmax=1, color='lightgray', alpha=.8,
               lw=3,)
    
    plot_Rs = []
    np.random.seed(11)
    x_jitt = np.random.uniform(low=-.15, high=.15,
                               size=len(sub_ids))

    for i, s in enumerate(sub_ids):
        if isinstance(sub_Rs[i], float) and ~np.isnan(sub_Rs[i]):
            plot_Rs.append(sub_Rs[i])
            if s[:3] == 'BER':
                hatch='//'
                clr=BER_clr
            elif s[:3] == 'DUS':
                    hatch=''
                    clr=DUS_clr
            ax.scatter(x_jitt[i], sub_Rs[i], s=sub_samples[i]*25,
                       color=clr, edgecolor=None,
                       alpha=.5, hatch=hatch, )
    
    # ax.axhline(np.nanmean(plot_Rs), xmin=0, xmax=1,
    #            color='darkblue', lw=3, alpha=.7,
    #            label='Mean')
    # viol_parts = ax.violinplot(
    #     [plot_Rs], positions=[0,],
    #     showmeans=False, showextrema=False,
    #     showmedians=False, vert=True,)
    # for pc in viol_parts['bodies']:
    #     pc.set_facecolor('blue')
    #     pc.set_edgecolor(None)
    #     pc.set_alpha(.3)
    
    ax.set_xlim(-.3, .5)
    ax.set_ylabel('Individual Pearson R',
                  size=fs, weight='bold')
    ax.set_yticks([-1, -0.5, 0, .5, 1], size=fs,)
    ax.set_yticklabels(['-1', '-0.5', '0', '0.5', '1'], size=fs,)
    ax.set_ylim(-1.5, 1.1)
    ax.set_xticks([], size=fs)
    ax.scatter(2, 2, s=300, hatch='//', alpha=.5,
           color=BER_clr, label='BER')
    ax.scatter(2, 2, s=300, hatch='', alpha=.5,
           color=DUS_clr, label='DUS')
    ax.scatter(2, 2, s=10*25, alpha=.7,
           color='darkgray', label='n=10')
    ax.scatter(2, 2, s=20*25, alpha=.7,
           color='darkgray', label='n=20')
    
    ax.legend(frameon=True, ncol=1, fontsize=fs-4,
              loc='lower right',)

    ax.tick_params(axis='both', labelsize=fs, size=fs)
    ax.spines[['right', 'top', 'bottom']].set_visible(False)

    return ax
    

def plot_holdout_per_sub(
    y_true_list, y_pred_list, trace_ids_list,
    subs_in_holdout, mc_labels=['0', '1', '2', '3'],
    fs=14, r_method=pearsonr,
    to_save=False, fname=None, add_folder=None, to_show=False,
):
    (sub_Rs,
     sub_samples,
     sub_ids,
     preds_per_sub,
     trues_per_sub) = get_indiv_holdout_results(
        y_true_list, y_pred_list, trace_ids_list,
        subs_in_holdout, r_method=r_method,
    )

    fig, axes = plt.subplots(int(len(subs_in_holdout) / 2), 2,
                             figsize=(8, 12),
                            #  sharex=True, sharey=True,
                             )
            
    axes = axes.flatten()

    for n, sub in enumerate(subs_in_holdout):
        
        axes[n].set_title(f'Subject {sub}: R: {sub_Rs[n]}',
                          weight='bold', fontsize=fs, c='darkgray')

        jitt = np.random.uniform(low=-.2, high=0.2,
                                 size=len(trues_per_sub[sub]))
        jitt2 = np.random.uniform(low=-.2, high=0.2,
                                  size=len(trues_per_sub[sub]))
        axes[n].scatter(preds_per_sub[sub]+jitt2,
                        trues_per_sub[sub]+jitt)
        
        axes[n].set_xticks(range(4))
        axes[n].set_yticks(range(4))
        axes[n].spines[['right', 'top']].set_visible(False)
        axes[n].set_yticklabels(mc_labels, weight='bold', fontsize=fs)
        axes[n].set_xticklabels(mc_labels, weight='bold', fontsize=fs)
        axes[n].set_xlim(-.2, 3.2)
        axes[n].set_ylim(-.2, 3.2)

    for n in range(0, len(subs_in_holdout), 2):
        axes[n].set_ylabel('True', weight='bold', fontsize=fs)
    for n in [8, 9]:
        axes[n].set_xlabel('Predicted', weight='bold', fontsize=fs)

    
    plt.tight_layout(pad=.5,)

    if to_save:
        path = os.path.join(
            find_onedrive_path('figures'),
            'prediction'
        )
        if add_folder: path = os.path.join(path, add_folder)
        plt.savefig(os.path.join(path, fname), dpi=300, facecolor='w',)
        print(f'Saved Fig "{fname}" in {path}')
        plt.close()
    else:
        plt.show()


def get_indiv_holdout_results(
    y_true_list, y_pred_list, trace_ids_list,
    subs_in_holdout, r_method,
):
    sub_Rs, sub_samples, sub_ids = [], [], []
    preds_per_sub, trues_per_sub = {}, {}

    for sub in subs_in_holdout:
        sub_ids.append(sub)
        
        # sub_traces = [t for t in all_traceids if t.startswith(sub)]
        preds_per_sub[sub] = [v for t, v in zip(trace_ids_list, y_pred_list)
                     if t.startswith(sub)]
        trues_per_sub[sub] = [v for t, v in zip(trace_ids_list, y_true_list)
                     if t.startswith(sub)]
        # get indiv correlation
        if len(trues_per_sub[sub]) == 1:
            R = 'N/A'
        else:
            R, R_p = r_method(preds_per_sub[sub], trues_per_sub[sub])
            R = round(R, 3)
        sub_Rs.append(R)
        sub_samples.append(len(trues_per_sub[sub]))

    return sub_Rs, sub_samples, sub_ids, preds_per_sub, trues_per_sub


def plot_ft_importances(
    fitted_clf, ft_names, model_name,
    to_save=True, ADD_FIG_PATH='v2',
):
    # get random forest importances using mean decrease in impurity
    # https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html
    importances = fitted_clf.feature_importances_
    std = np.std([tree.feature_importances_ for tree
                  in fitted_clf.estimators_], axis=0)
    forest_importances = Series(importances, index=ft_names)

    plot_ft_labels = [
        'normed RMS (trace)',
        'entropy (trace)',
        'jerkiness (trace)',
        'ITI (coefvar)',
        'ITI (decrement)',
        'tap-RMS (mean)',
        'tap-RMS (coefvar)',
        'impact-RMS (mean)',
        'impact-RMS (coefvar)',
        'impact-RMS (decrement)',
        'raise-velocity (mean)',
        'raise-velocity (coefvar)',
        'tap-entropy (coefvar)',
        'tap-entropy (decrement)'
    ]

    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=std, ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_xticklabels(plot_ft_labels)
    ax.set_ylabel("Mean decrease in impurity (a.u.)")
    fig.tight_layout()

    if to_save:
        fname = f'ftImportances_{model_name}'
        path = os.path.join(
            find_onedrive_path('figures'),
            'prediction'
        )
        if ADD_FIG_PATH: path = os.path.join(path, ADD_FIG_PATH)
        plt.savefig(os.path.join(path, fname), dpi=300, facecolor='w',)
        print(f'Saved Fig "{fname}" in {path}')
        plt.close()
    else:
        plt.show()

