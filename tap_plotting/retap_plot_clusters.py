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
from os import makedirs
import datetime as dt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Import own functions
from retap_utils import utils_dataManagement as utilsDatamng
from tap_extract_fts.tapping_postFeatExtr_calc import z_score_array, normalize_var_fts

# TODO: ANNOTATE ANALYSIS PARAMETERS IN LOWER RIGHT CORNER

def get_kMeans_clusters(
    X,
    n_clusters=2,
    use_pca=True,
    to_zscore=False,
    to_norm=False,
    random_state=27,
):
    """
    Extract clusters (y_clusters) based on inserted value in X

    Input:
        - X: array of shape (n-samples, n-features)
    
    Returns:
        - y_clusters (array): shape (n-samples, 1)
        - cluster_centroids
        - X_pca: if pca is used, otherwise None
    """
    if to_zscore:
        for i_ft in range(X.shape[1]):
            X[:, i_ft] = z_score_array(X[:, i_ft])
    elif to_norm:
        for i_ft in range(X.shape[1]):
            X[:, i_ft] = normalize_var_fts(X[:, i_ft])

    if use_pca:
        pca = PCA(2)
        X_pca = pca.fit_transform(X)
        X = X_pca
        plot_axes = X_pca
    else:
        plot_axes = X

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state
    )
    y_clust_labels = kmeans.fit_predict(X)
    cluster_centroids = kmeans.cluster_centers_

    return y_clust_labels, cluster_centroids, plot_axes
    

def plot_cluster_kMeans(
    X,
    y,
    n_clusters=2,
    use_pca=True,
    z_score=False,
    alt_labels=['feature 1', 'feature 2'],
    random_state=27,
    figsave_name: str='',
    figsave_dir: str='',
    show: bool=False
):
    """
    Plotting cluster with labeling on true labels
    """
    y_clust_labels, centroids, plot_axes = get_kMeans_clusters(
        X=X, n_clusters=n_clusters,
        use_pca=use_pca, z_score=z_score,
        random_state=random_state,
    )
    

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
            plot_axes[i_row, 0], plot_axes[i_row, 1],
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

    if use_pca: xlab, ylab = ['PCA-1', 'PCA-2']
    else: xlab, ylab = alt_labels

    ax.set_xlabel(xlab, fontsize=18)
    ax.set_ylabel(ylab, fontsize=18)
    ax.set_title(
        'kMeans Clustering 10-seconds of Finger Tapping',
        fontsize=20
    )

    plt.tight_layout()

    if len(figsave_name) > 1:
        if not exists(figsave_dir): makedirs(figsave_dir)
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

    
    