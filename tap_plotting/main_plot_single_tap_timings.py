"""
Visualizing single tap timings, FIGURE 1C
"""

# import functions and packages
import numpy as np
from os.path import exists, join
from os import makedirs, listdir
import matplotlib.pyplot as plt
# from matplotlib import cm
import json

from retap_utils.utils_dataManagement import (
    find_onedrive_path, get_local_proj_dir,
    load_class_pickle
)


def get_colors(scheme='access_colors_PaulTol'):
    """
    if scheme is 'access_colors_PaulTol' Paul Tol's
        colorscheme for accessible colors is returned,
    if scheme is 'contrast_duo', two contrasting colors
        are returned
    """
    cmap_json = join(get_local_proj_dir(), 'code',
                     'updrsTapping_repo', 'tap_plotting',
                     'color_schemes.json')

    with open(cmap_json, 'r') as json_data:
        schemes = json.load(json_data)

    cmap = schemes[scheme]

    return cmap


def plot_single_tap(
    acc_arr, fs, impacts, timing_lists,
    figsave_dir,
    figsave_name='FIG1C_singleTap_featTimings',
    save_as_pdf=False,
):
    """
    Plots overview of selected blocks and main axes
    """
    if not exists(figsave_dir):
        makedirs(figsave_dir)

    print(f'plotting {figsave_name}...')

    fontsize = 16
    sec_p_xtick = .25  # seconds per xticks
    ymin, ymax = -1.5, 1.5

    t_labels = ['start-raise', 'fastest-raise', 'stop-raise', 'start-lower', 
                'fastest-lower', 'impact', 'end-lower']
    del(t_labels[5])

    cmap = get_colors()
    cmap = list(cmap.values())
    duo_clrs = get_colors(scheme='contrast_duo')

    START_TAP, STOP_TAP = 7, 10

    
    fig, ax = plt.subplots(1, 1, figsize=(16, 6))

    x_start = impacts[START_TAP]
    x_stop = impacts[STOP_TAP]
    
    ax.plot(acc_arr[0], c='gray', alpha=.6,)
    ax.plot(acc_arr[1], c='darkgray', alpha=.6,)
    ax.plot(acc_arr[2], c='k', alpha=.6,)

    ax.axhline(y=0,
            #    xmin=x_start-25, xmax=x_stop+25, 
               color='k', alpha=.5, lw=1, ls='dotted',)

    for x_impact in impacts:
        ax.axvline(x=x_impact, ymin=ymin, ymax=ymax, c='gray',
                   alpha=.2, lw=5, ls='--', label='impact')

    for timings in timing_lists[START_TAP:STOP_TAP]:
        timings = list(timings)
        del(timings[5])
        # # plot every line vertically
        # for t_i, t in enumerate(timings):
        #     if np.isnan(t): continue
        #     ax.axvline(x=t, ymin=0, ymax=ymax,
        #                color=cmap[t_i],
        #                alpha=.7, lw=2,
        #                label=t_labels[t_i])
        # plot raise and lower
        ax.fill_betweenx(y=np.arange(ymin - 1, ymax + 1),
                         x1=timings[0], x2=timings[2],
                         color=duo_clrs[0], alpha=.3,
                         label='finger opening',)
        ax.axvline(x=timings[1], ymin=ymin, ymax=ymax, color=duo_clrs[0],
                   alpha=.7, lw=2, label='max opening velocity')
        ax.fill_betweenx(y=np.arange(ymin - 1, ymax + 1),
                         x1=timings[3], x2=timings[5],
                         color=duo_clrs[1], alpha=.3,
                         label='finger closing')
        ax.axvline(x=timings[4], ymin=ymin, ymax=ymax,
                   color=duo_clrs[1], alpha=.7,
                   lw=2, label='max closing velocity')

    xticks = np.arange(0, len(acc_arr[0]), fs*sec_p_xtick)
    ax.set_xticks(xticks)
    ax.set_xticklabels(np.around(xticks/fs, 2), fontsize=fontsize)
    ax.set_yticks([-1, 0, 1])
    ax.set_yticklabels([-1, 0, 1], fontsize=fontsize)
    ax.set_xlabel(f'Time (seconds)', fontsize=fontsize, weight='bold',)
    ax.set_ylabel('Acceleration (g)', fontsize=fontsize, weight='bold',)
    
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(x_start - 25, x_stop + 25)

    ax.spines[['right', 'top', 'bottom']].set_visible(False)  # 

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    legend = plt.legend(
        by_label.values(), by_label.keys(),
        frameon=False, fontsize=fontsize,
        ncol=5, loc='lower center',
        bbox_to_anchor=(.5, -.5)
    )

    plt.tick_params(axis='both', labelsize=fontsize,
                    size=fontsize,)
    plt.tight_layout()
    if save_as_pdf:
        figsave_name += '.pdf'
        fig_format = 'pdf'
    else:
        fig_format = 'png'
        figsave_name += '.png'

    plt.savefig(join(figsave_dir, figsave_name),
                format=fig_format,
                dpi=450, facecolor='w',)
    plt.close()



### RUN FROM COMMAND LINE

if __name__ == '__main__':
    # import sys to access inserted variables
    import sys
    # import original classes to load feature class-pickle
    from tap_extract_fts.main_featExtractionClass import FeatureSet, singleTrace

    """
    function is called in terminal (from repo folder):

    python -m tap_plotting.main_plot_single_tap_timings
    """
    # VARIABLES TO SELECT TAP DATA TO PLOT
    ftClass_file =  "ftClass_ALL.P"
    trace_to_plot = 'BER056_M1S0_L_1'   # 'BER026_M0S0_R_0
    
    deriv_path = join(get_local_proj_dir(), 'data', 'derivatives')
    figsave_dir = find_onedrive_path('figures')

    ftClass = load_class_pickle(join(deriv_path, ftClass_file))
    
    assert trace_to_plot in ftClass.incl_traces, 'trace not in ftClass'

    tapClass = getattr(ftClass, trace_to_plot)
    if 'tapDict' in vars(tapClass.fts).keys():
        timing_lists=tapClass.fts.tapDict
    else: timing_lists=tapClass.fts.tap_lists

    plot_single_tap(acc_arr=tapClass.acc_sig, fs=tapClass.fs,
                    impacts=tapClass.fts.impacts,
                    timing_lists=timing_lists,
                    figsave_dir=figsave_dir,
                    save_as_pdf=True,)