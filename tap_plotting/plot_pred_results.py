


# import function
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score as kappa  
from scipy.stats import spearmanr

from tap_predict import retap_cv_models as cv_models
from retap_utils.utils_dataManagement import find_onedrive_path


def plot_confMatrix_scatter(
    y_true, y_pred,
    R=None, K=None, CM=None,
    mc_labels=['0', '1', '2', '3-4'],
    to_save=False, fname=None,
    to_show=False,
):
    mean_pen, std_pen, _ = cv_models.get_penalties_from_conf_matr(CM)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4),
                            gridspec_kw={'width_ratios': [2, 1]},)
    fs=14

    jitt = np.random.uniform(low=-.2, high=0.2, size=len(y_true))
    jitt2 = np.random.uniform(low=-.2, high=0.2, size=len(y_true))

    axes[0].scatter(y_true+jitt, y_pred+jitt2,
                    alpha=.5,)
    axes[0].set_xlabel('True Tap Score', weight='bold', fontsize=fs)
    axes[0].set_ylabel('Predicted Tap Score', weight='bold', fontsize=fs)
    axes[0].set_xticks(range(4))
    axes[0].set_xticklabels(mc_labels, weight='bold', fontsize=fs)
    axes[0].set_yticks(range(4))
    axes[0].set_yticklabels(mc_labels, weight='bold', fontsize=fs)
    axes[0].spines[['right', 'top']].set_visible(False)


    ### MAKE HEATMAP of conf matrix

    # add missing data in CM table
    for l in mc_labels:
        if l not in CM.index:
            CM.loc[l] = [0] * len(mc_labels)
    CM = CM.sort_index()

    im = axes[1].imshow(CM.values)

    # Show all ticks and label them with the respective list entries
    axes[1].xaxis.tick_top()
    axes[1].set_xticks(np.arange(len(mc_labels)))
    axes[1].set_yticks(np.arange(len(mc_labels)))
    axes[1].set_xticklabels(mc_labels, weight='bold', fontsize=fs)
    axes[1].set_yticklabels(mc_labels, weight='bold', fontsize=fs, )
    axes[1].xaxis.set_label_position('top')
    axes[1].set_xlabel('Predicted UPDRS Tap Score', weight='bold', fontsize=fs, )
    axes[1].set_ylabel('True UPDRS Tap Score', weight='bold', fontsize=fs)

    # Loop over data dimensions and create text annotations.
    for i in range(len(mc_labels)):
        for j in range(len(mc_labels)):
            value = CM.values[i, j]
            if value > 30: txt_col ='k'
            else: txt_col = 'w'
            text = axes[1].text(
                j, i, value, weight='bold', fontsize=fs,
                ha="center", va="center", color=txt_col,
      )
    
    # Metrics in title
    # axes[1].set_title(
    #     (f'Mean Prediction Error: {mean_pen.round(2)} (sd: {std_pen.round(2)})'
    #     f'\nR: {R.round(3)}, Kappa: {K.round(2)}'),
    #     # xy=(.5, .2), ha='center', va='top',
    #     # xycoords='axes fraction',
    #     fontsize=fs,
    # )


    ### MAKE TABLE of conf matrix
    # cell_text = []
    # for row in range(len(CM)):
    #     cell_text.append(CM.iloc[row])
    # rcolors = plt.cm.BuPu(np.full(len(mc_labels), 0.1))
    # ccolors = plt.cm.BuPu(np.full(len(mc_labels), 0.1))
    # plot_table = axes[1].table(
    #     cellText=cell_text,
    #     colLabels=mc_labels,
    #     rowLabels=mc_labels,
    #     rowColours=rcolors,
    #     rowLoc='right',
    #     colColours=ccolors,
    #     loc='center',
    # )
    # plot_table.scale(1, 1.5)
    # axes[1].annotate(
    #     'Pred. Scores', xy=(.1, .73), xycoords='axes fraction',
    #     weight='bold', fontsize=fs,)
    # axes[1].annotate(
    #     'True Scores', xy=(-.3, .3), xycoords='axes fraction',
    #     rotation=90, weight='bold', fontsize=fs,)

    # ANNOTATE METRICS
    axes[1].annotate(
        (f'Mean Prediction Error: {mean_pen.round(2)} (sd: {std_pen.round(2)})'
        f'\nR: {R.round(3)}, Kappa: {K.round(2)}'),
        xy=(.5, -.3), ha='center', va='top',
        xycoords='axes fraction',
        fontsize=fs - 2,)
    
    # axes[1].axis('off')

    plt.tight_layout(w_pad=.5)

    if to_save:
        path = os.path.join(
            find_onedrive_path('figures'),
            'prediction'
        )
        plt.savefig(os.path.join(path, fname), dpi=150, facecolor='w',)
        print(f'Saved Fig "{fname}" in {path}')
    if to_show: plt.show()
    else: plt.close()