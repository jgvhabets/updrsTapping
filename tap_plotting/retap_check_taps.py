"""
Visualise ReTap Detected Taps

uses class resulting main_featExtractionClass(), which
contain one class with attribute fts, per 10-sec trace
"""

# Import public packages and functions
import numpy as np
import matplotlib.pyplot as plt
from os.path import join, exists
from os import makedirs

# Import own fucntions
from retap_utils.utils_dataManagement import get_local_proj_dir, load_class_pickle

# only when alterantives methods should be plotted:
# from tap_load_data.tapping_impact_finder import find_impacts
# from tap_extract_fts.tapping_featureset import signalvectormagn
# from tap_load_data.tapping_preprocess import find_main_axis

def plot_detected_taps(
    ftClass, fontsize=16,
):

    savepath = join(
        get_local_proj_dir(),
        'figures',
        'tap_find_check', 'submission'
    )
    if not exists(savepath): makedirs(savepath)

    for trace in ftClass.incl_traces:

        testrun = getattr(ftClass, trace)

        accsig = testrun.acc_sig
        taps = [t[0] for t in testrun.fts.tap_lists]

        x_samples = np.arange(accsig.shape[1])

        plt.plot(accsig.T, alpha=.3, label=['X', 'Y', 'Z'])

        plt.scatter(taps, [3] * len(taps), label='tap-starts')


        ## plotting of alterantive tap-detection methods
        # svm = signalvectormagn(accsig)
        # mainax = find_main_axis(accsig, method='minmax')
        # impacts = find_impacts(accsig[mainax], 250)
        # svmimpacts = find_impacts(svm, 250)
        # plt.plot(svm, alpha=.4)
        # plt.scatter(impacts, [4] * len(impacts))
        # plt.scatter(svmimpacts, [3.5] * len(svmimpacts))
        

        plt.ylabel('Acceleration (g)', fontsize=16,)
        plt.xlabel('Time (sec)', fontsize=16,)

        xtick_res_sec = 2 # resolution of xtick labels
        plt.xticks(
            x_samples[::250 * xtick_res_sec],
            labels=x_samples[::250 * xtick_res_sec] / 250)  # divide by fs for second-convertion

        # plt.legend(
        #     loc='lower center',ncol=4)

        plt.title(f'{trace}   ({testrun.tap_score})')

        plt.savefig(
            join(savepath, f'{trace}_tap_check.pdf'),
            format='pdf',
            dpi=450, facecolor='w',
        )
        plt.close()


### RUN FROM COMMAND LINE

if __name__ == '__main__':

    import sys

    # import original classes to load feature class-pickle
    from tap_extract_fts.main_featExtractionClass import FeatureSet, singleTrace

    """
    function is called in terminal (from repo folder):

    python -m tap_plotting.retap_check_taps "ftClass_ALL.P"
    
        - -m needs to be added bcs the file is called within a method/folder from current work dir
        - second argument is filename of pickle saved class
        - if third argument is given, this is the ft-list to include
            (if not given it is extracted by default in sort_fts_on_tapScore()) 
    """
    assert len(sys.argv) == 2, ('Define at least second variable with pickle-filename')

    deriv_path = join(
        get_local_proj_dir(),
        'data', 'derivatives')

    ftClass_file = sys.argv[1]
    ftClass = load_class_pickle(
        join(deriv_path, ftClass_file))

    plot_detected_taps(ftClass)