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

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Import own functions
from retap_utils import utils_dataManagement as utilsDatamng

# TODO: ANNOTATE ANALYSIS PARAMETERS IN LOWER RIGHT CORNER

def plot_cluster_kMeans(
    X, y,
    n_clusters=2,
    use_pca=True,
    random_state=27,
    figsave_name: str='',
    figsave_dir: str='',
    show: bool=False
):
    if not exists(figsave_dir): mkdir(figsave_dir)

    pca = PCA(2)
    X_pca = pca.fit_transform(X)
    if use_pca: X = X_pca

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state
    )
    y_clust_labels = kmeans.fit_predict(X)
    centroids = kmeans.cluster_centers_

    score_cols = {
        0: 'green',
        1: 'lightgreen',
        2: 'orange',
        3: 'red',
        4: 'purple'
    }

    cl_markers = ['o', 'x', '*', '+', '^']

    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    s = 75

    for i_row in np.arange(X.shape[0]):  # one X_row are all features from one trace

        score = int(y[i_row])
        color = score_cols[score]

        pred_clust = y_clust_labels[i_row]
        marker = cl_markers[pred_clust]

        ax.scatter(
            X_pca[i_row, 0], X_pca[i_row, 1],
            label=f'Cluster-{pred_clust}, Tap-Score {score}',
            s=s, color=color, alpha=.7,
            marker=marker,
        )


    for c in range(centroids.shape[0]):
        ax.scatter(
            centroids[c, 0], centroids[c, 1],
            edgecolor='k', s=s + 50, fc='w',
            label='Cluster centers'
        )

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(
        by_label.values(), by_label.keys(),
        frameon=False, fontsize=16,
        ncol=1,
        loc='upper left',
        bbox_to_anchor=[1.01, .95]
    )

    ax.set_xlabel('PCA-1', fontsize=18)
    ax.set_ylabel('PCA-2', fontsize=18)
    ax.set_title(
        'kMeans Clustering 10-seconds of Finger Tapping',
        fontsize=20
    )

    plt.tight_layout()

    if len(figsave_name) > 1:
        plt.savefig(
            join(figsave_dir, figsave_name),
            dpi=150, facecolor='w',
        )
    if show: plt.show()
    plt.close()


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
    # assert len(sys.argv) > 1, ('Define at least second variable with pickle-filename')

    # # import original classes to load feature class-pickle
    # from tap_extract_fts.main_featExtractionClass import FeatureSet, singleTrace

    # deriv_path = join(
    #     utilsDatamng.get_local_proj_dir(),
    #     'data', 'derivatives')

    # ftClass_file = sys.argv[1]
    # ftClass = utilsDatamng.load_class_pickle(
    #     join(deriv_path, ftClass_file))
    
    # ft_to_plot = 'jerkiness_smooth'  # tapRMSnrm, raise_velocity, intraTapInt, jerkiness, jerkiness_smooth
    # metrics = 'mean', 'coefVar', 'IQR', 'decr', 'slope',
    # fts_include = [f'{m}_{ft_to_plot}' for m in metrics]
    
    # if len(sys.argv) == 2:  # no fts_include defined, take default
    #     sorted_feats, ft_list = sort_fts_on_tapScore(ftClass=ftClass, fts_include=fts_include)
    # elif len(sys.argv) == 3:  # fts_include defined
    #     sorted_feats, ft_list = sort_fts_on_tapScore(ftClass=ftClass, fts_include=sys.argv[2])
    
    
    # fig_fname = (
    #     f'{dt.date.today().year}{dt.date.today().month}'
    #     f'{dt.date.today().day}_kClusters_'
    #     f'{ft_to_plot}_{sys.argv[1].split(".")[0]}_'
    # )

    # plot_xxx(
    #     fts_include=ft_list,
    #     sorted_feat_dict=sorted_feats,
    #     # plot_title='',
    #     figsave_name=fig_fname,
    #     figsave_dir=join(
    #         utilsDatamng.find_onedrive_path('figures'),
    #         'fts_boxplots',
    #     ),
    #     show=False
    # )

    
    