"""
Visualise ReTap Tapping-Features

uses class resulting main_featExtractionClass(), which
contain one class with attribute fts, per 10-sec trace
"""

# Import public packages and functions
import sys
import numpy as np
import matplotlib.pyplot as plt
from itertools import compress
from os.path import join, exists
from os import mkdir
import datetime as dt

# Import own functions
from retap_utils import utils_dataManagement as utilsDatamng

def sort_fts_on_tapScore(
    ftClass,
    fts_include: list = [
        'mean_tapRMSnrm', 'coefVar_tapRMSnrm', 'IQR_tapRMSnrm', 'decr_tapRMSnrm', 'slope_tapRMSnrm', 
        'mean_raise_velocity', 'coefVar_raise_velocity', 'IQR_raise_velocity', 'decr_raise_velocity', 'slope_raise_velocity',
        'mean_intraTapInt', 'coefVar_intraTapInt', 'IQR_intraTapInt', 'decr_intraTapInt', 'slope_intraTapInt',
        'mean_jerkiness', 'coefVar_jerkiness', 'IQR_jerkiness', 'decr_jerkiness', 'slope_jerkiness',
        'mean_jerkiness_smooth', 'coefVar_jerkiness_smooth', 'IQR_jerkiness_smooth', 'decr_jerkiness_smooth', 'slope_jerkiness_smooth'
    ],
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
    
    Returns:
        - feat_dict_out (dict): containing one dict
            per features; with features per
            updrs tapping subscore
            
    """
    feat_dict_out = {}
    for ft_sel in fts_include:

        ft_per_score = {}
        for s in np.arange(5): ft_per_score[s] = []  # for every possible updrs subscore one list

        for trace in ftClass.incl_traces:
            
            traceClass = getattr(ftClass, trace)
            # do not include traces without detected taps
            if len(traceClass.fts.tapDict) == 0:
                print(
                    f'({ft_sel})\t0 taps available for {trace}'
                    f' (tap-subscore: {traceClass.tap_score})')
                continue

            tap_score = traceClass.tap_score

            try:
                ft_score = getattr(traceClass.fts, ft_sel)
            except AttributeError:
                print(f'{trace}\nattributes: {vars(traceClass.fts).keys()}')
                raise AttributeError

            if ~np.isnan(ft_score): ft_per_score[tap_score].append(ft_score)

        feat_dict_out[ft_sel] = ft_per_score

    return feat_dict_out, fts_include



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
    fts_include: list,
    sorted_feat_dict: dict,
    plot_title: str='',
    figsave_name: str='',
    figsave_dir: str='',
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
        fts_scores = sorted_feat_dict[ft_sel]
        boxdata = [fts_scores[s] for s in fts_scores.keys()]
        boxdata = clean_list_of_lists(boxdata)
        axes[row].boxplot(boxdata)  # boxplot function takes list of lists

        tick_Ns = [len(l) for l in boxdata]
        xlabels = [
            f'{i}  (n={tick_Ns[i]})' for i in np.arange(0, 5)
        ]
        axes[row].set_xticklabels(
            xlabels, fontsize=18, ha='center',)
        axes[row].set_xlabel('UPDRS tapping sub score', fontsize=18)
        axes[row].set_ylabel(ft_sel, fontsize=18)

    if len(plot_title) > 1:
        plt.suptitle(
            f'{plot_title}',
            size=24, x=.5, y=.92, ha='center',
        )
    
    if figsave_name:
        
        plt.savefig(
            join(figsave_dir, figsave_name),
            dpi=150, facecolor='w',
        )
    if show: plt.show()
    if not show: plt.close()


### RUN FROM COMMAND LINE

if __name__ == '__main__':
    """
    function is called in terminal with:

    python -m tap_plotting.retap_visualise_fts "ftClass_ALL.P"
    
        - -m needs to be added bcs the file is called within a method/folder from current work dir
        - second argument is filename of pickle saved class
        - if third argument is given, this is the ft-list to include
            (if not given it is extracted by default in sort_fts_on_tapScore()) 
    """
    assert len(sys.argv) > 1, ('Define at least second variable with pickle-filename')

    # import original classes to load feature class-pickle
    from tap_extract_fts.main_featExtractionClass import FeatureSet, singleTrace

    deriv_path = join(
        utilsDatamng.get_local_proj_dir(),
        'data', 'derivatives')

    ftClass_file = sys.argv[1]
    ftClass = utilsDatamng.load_class_pickle(
        join(deriv_path, ftClass_file))
    
    ft_to_plot = 'jerkiness_smooth'  # tapRMSnrm, raise_velocity, intraTapInt, jerkiness, jerkiness_smooth
    metrics = 'mean', 'coefVar', 'IQR', 'decr', 'slope',
    fts_include = [f'{m}_{ft_to_plot}' for m in metrics]
    
    if len(sys.argv) == 2:  # no fts_include defined, take default
        sorted_feats, ft_list = sort_fts_on_tapScore(ftClass=ftClass, fts_include=fts_include)
    elif len(sys.argv) == 3:  # fts_include defined
        sorted_feats, ft_list = sort_fts_on_tapScore(ftClass=ftClass, fts_include=sys.argv[2])
    
    
    fig_fname = (
        f'{dt.date.today().year}{dt.date.today().month}'
        f'{dt.date.today().day}_ftBoxplot_'
        f'{ft_to_plot}_{sys.argv[1].split(".")[0]}_'
    )

    plot_boxplot_feats_per_subscore(
        fts_include=ft_list,
        sorted_feat_dict=sorted_feats,
        # plot_title='',
        figsave_name=fig_fname,
        figsave_dir=join(
            utilsDatamng.find_onedrive_path('figures'),
            'fts_boxplots',
        ),
        show=False
    )

    
    