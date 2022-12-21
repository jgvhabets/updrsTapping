"""
Plotting functions for classifying trials
"""

# import public functions
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import (
    RocCurveDisplay,  ConfusionMatrixDisplay,
    roc_curve, accuracy_score,
    roc_auc_score
)

# import own functions
from retap_utils.plot_helpers import remove_duplicate_legend



def plot_ROC_AUC_confMatrices_for_folds(
    y_true_dict, y_proba_dict,
    plot_thresholds=[.4, .5, .6],
    roc_title: str='Receiver Operator Curve (per fold)',
    incl_mean_ROC: bool = False,
):

    # optimal threshold for PUDRS <= 1 = .58 (both LDA and LogRegr) or .6
    # optimal threshold for PUDRS >= 3 = .1 - .2 (both LDA and LogRegr)
    clrs = ['blue', 'purple', 'orange', 'magenta']
    mrkrs = ['o', '*', 'd', 'p']

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    if incl_mean_ROC:
        tprs = []
        mean_fpr = np.linspace(0, 1, 100)
        aucs = []

    for f in range(len(y_true_dict)):
        roc_y_true = y_true_dict[f]
        roc_y_probs = y_proba_dict[f][:, 1]

        fpr, tpr, thresholds = roc_curve(roc_y_true, roc_y_probs)

        if incl_mean_ROC:  # interpolate ROC over 100 samples
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(roc_auc_score(roc_y_true, roc_y_probs))

        for th_i, thr in enumerate(plot_thresholds):
            t_nearest = np.argmin(abs(thresholds - thr))
            ax.scatter(
                fpr[t_nearest], tpr[t_nearest],
                color=clrs[f], marker=mrkrs[th_i],

            )
            # plot empty scatter for gray legend
            ax.scatter(
                [], [],
                color='gray', marker=mrkrs[th_i],
                label=f'threshold {thr}'
            )
        # Plot ROC
        display = RocCurveDisplay.from_predictions(
            y_true=roc_y_true, y_pred=roc_y_probs,
            ax=ax,
            name=f'AUC ROC - fold {f}',
            alpha=.4, color=clrs[f],
        )

        # PLOT CONFUSION MATRICES
        fig2, axes2 = plt.subplots(1, len(plot_thresholds),
                                   figsize=(12, 4))

        for n_th, thr in enumerate(plot_thresholds):
        
            ConfusionMatrixDisplay.from_predictions(
                y_true=roc_y_true,
                y_pred=roc_y_probs > thr,
                ax=axes2[n_th],
            )
            axes2[n_th].set_ylabel('True label (UPDRS <= 1)')
            axes2[n_th].set_xlabel(f'Predicted label (thr={thr})')
            axes2[n_th].set_title(
                f'Accuracy: {round(accuracy_score(roc_y_true, roc_y_probs > thr), 2)}')
        fig2.tight_layout()
        fig2.show()
    
    if incl_mean_ROC:
        # calculate interpolated mean and std dev
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        # calculate mean AUC
        mean_auc = np.round(np.mean(aucs), 2)
        print(aucs)
        std_auc = np.round(np.std(aucs), 2)
        # plot mean and std dev
        ax.plot(
            mean_fpr, mean_tpr,
            color="b", lw=2, alpha=0.7,
            label=f'Mean ROC (AUC = {mean_auc} +/- {std_auc})',
        )
        ax.fill_between(
            mean_fpr, tprs_lower, tprs_upper,
            color="grey", alpha=0.2,
            label='+/- 1 std. dev.',
        )

    ax.set_ylabel('True Positive Rate')
    ax.set_xlabel('False Positive Rate')
    ax.set_title(roc_title)

    legend_handles_labels = fig.gca().get_legend_handles_labels()
    handles, labels = remove_duplicate_legend(legend_handles_labels)
    ax.legend(handles, labels, frameon=False)

    plt.show()

