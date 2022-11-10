"""
Visualise ReTap Tapping-Features

uses class resulting main_featExtractionClass(), which
contain one class with attribute fts, per 10-sec trace
"""

# Import public packages and functions
import numpy as np
import matplotlib.pyplot as plt
from itertools import compress
from os.path import join, exists
from os import mkdir

# Import own fucntions
from tap_extract_fts.tapping_feat_calc import aggregate_arr_fts


def combineFeatsPerScore(
    ftClass: dict, fts_include, merge_method: str,
):
    """
    Merges and compares features of different
    tapping runs, classifies based on the cor-
    responding UPDRS tapping subscore

    Input:
        - ftClass (class): containing Classes
            (from main_featExtractionClass())
            with features and info per run
        - fts_include (list): define which
            features to include in analysis
        - merge_method (str): method to
            aggregate array-based features
            per run, most be from defined list
    
    Returns:
        feat_dict_out (dict): containing one dict
            per features; with features per
            updrs tapping subscore
            
    """
    merge_method_list = [
        'allin1', 'mean', 'sum', 'stddev', 'coefVar',
        'range', 'trend_slope', 'trend_R', 'median', 'variance',
    ]
    assert merge_method in merge_method_list, (
        f'merge_method: "{merge_method}" is not '
        f'in {merge_method_list}')

    feat_dict_out = {}
    for ft_sel in fts_include:

        ft_per_score = {}
        for s in np.arange(5): ft_per_score[s] = []

        for trace in ftClass.incl_traces:
            
            traceClass = getattr(ftClass, trace)
            tap_score = traceClass.tap_score
            ft_score = getattr(traceClass.fts, ft_sel)

            if type(ft_score) != np.ndarray:  #float or np.float_
                ft_per_score[tap_score].append(ft_score)
            
            elif type(ft_score) == np.ndarray:

                if ft_score.size == 0: continue

                if merge_method == 'allin1':
                    if np.isnan(ft_score).any():
                        ft_score[~np.isnan(ft_score)]
                    ft_per_score[tap_score].extend(ft_score)  # all in one big list

                else:
                    ft_per_score[tap_score].append(
                        aggregate_arr_fts(
                            method=merge_method,
                            arr=ft_score
                        )
                    )

        feat_dict_out[ft_sel] = ft_per_score

    return feat_dict_out



def clean_list_of_lists(dirty_lists):
    """
    Remove nans from a list of lists
    """
    clean_lists = []
    
    for templist in dirty_lists:
            sel = ~np.isnan(templist)
            clean_lists.append(
                list(compress(templist, sel))
            )

    return clean_lists


def plot_boxplot_feats_per_subscore(
    fts_include: list, featDict: dict,
    merge_method: str, plot_title: str='',
    figsave_name: str='', figsave_dir: str='',
    show: bool=False
):
    """
    Plots boxplots of tapping features which
    are ordered per updrs subscore priorly.
    
    Input:
        - fts_include (list): list with feature names
        - fts_per_score (list): list containing
            feature-arrays per
    """
    if not exists(figsave_dir): mkdir(figsave_dir)

    fig, axes = plt.subplots(
        len(fts_include), 1, figsize=(24, 16)
    )

    for row, ft_sel in enumerate(fts_include):
        fts_scores = featDict[ft_sel]
        boxdata = [fts_scores[s] for s in fts_scores.keys()]
        boxdata = clean_list_of_lists(boxdata)
        axes[row].boxplot(boxdata)

        tick_Ns = [len(l) for l in boxdata]
        xlabels = [
            f'{i}  (n={tick_Ns[i]})' for i in np.arange(0, 5)
        ]
        axes[row].set_xticklabels(
            xlabels, fontsize=18, ha='center',)
        axes[row].set_xlabel('UPDRS tapping sub score', fontsize=18)
        axes[row].set_ylabel(ft_sel, fontsize=18)

    plt.suptitle(
        f'{plot_title} (merged by {merge_method})',
        size=24, x=.5, y=.92, ha='center',
    )
    
    if figsave_name:
        
        plt.savefig(
            join(figsave_dir, figsave_name),
            dpi=150, facecolor='w',
        )
    if show: plt.show()
    if not show: plt.close()