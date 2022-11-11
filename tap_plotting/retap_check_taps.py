"""
Visualise ReTap Detected Taps

uses class resulting main_featExtractionClass(), which
contain one class with attribute fts, per 10-sec trace
"""

# Import public packages and functions
import numpy as np
import matplotlib.pyplot as plt
from os.path import join, exists
from os import mkdir

# Import own fucntions
from retap_utils.utils_dataManagement import get_local_proj_dir

# only when alterantives methods should be plotted:
# from tap_load_data.tapping_impact_finder import find_impacts
# from tap_extract_fts.tapping_featureset import signalvectormagn
# from tap_load_data.tapping_preprocess import find_main_axis

def plot_detected_taps(
    ftClass,
):

    savepath = join(
        get_local_proj_dir(),
        'figures',
        'tap_find_check',
    )
    if not exists(savepath): mkdirs(savepath)


    for trace in ftClass.incl_traces:

        testrun = getattr(ftClass, trace)

        accsig = testrun.acc_sig
        taps = [t[0] for t in testrun.fts.tapDict]

        x_samples = np.arange(accsig.shape[1])
        plt.plot(accsig.T, alpha=.3, label=['x', 'y', 'z'])

        plt.scatter(taps, [3] * len(taps), label='tap-starts')


        ## plotting of alterantive tap-detection methods
        # svm = signalvectormagn(accsig)
        # mainax = find_main_axis(accsig, method='minmax')
        # impacts = find_impacts(accsig[mainax], 250)
        # svmimpacts = find_impacts(svm, 250)
        # plt.plot(svm, alpha=.4)
        # plt.scatter(impacts, [4] * len(impacts))
        # plt.scatter(svmimpacts, [3.5] * len(svmimpacts))
        

        plt.ylabel('Acceleration (g)')
        plt.xlabel('Time (s)')

        xtick_res_sec = 2 # resolution of xtick labels
        plt.xticks(
            x_samples[::250 * xtick_res_sec],
            labels=x_samples[::250 * xtick_res_sec] / 250)  # divide by fs for second-convertion

        plt.legend(loc='lower center', ncol=4)

        plt.title(trace)

        plt.savefig(
            join(savepath, f'{trace}_tap_check'),
            dpi=150, facecolor='w',
        )
        plt.close()