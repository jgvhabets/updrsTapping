'''Run UPDRS FingerTap Detection and Assessment functions'''

# Import public packages and functions
import numpy as np
from scipy.signal import resample
from itertools import compress
from typing import Any

# Import own functions
from tap_load_data.tapping_time_detect import updrsTapDetector
from tap_load_data.tapping_preprocess import run_preproc_acc, find_main_axis


def run_updrs_tap_finder(
    acc_arr, fs: int, already_preprocd: bool=True,
    orig_fs: Any=False,
):
    """
    Input:
        - acc_arr (array): tri-axial acc array
        - fs (int): sampling freq in Hz - in case of
            resampling: this is the wanted fs where the
            data is converted to.
        - already_preproc (bool): if True: preprocessing
            function is called
        - orig_fs: if preprocessing still has to be done,
            and the acc-signal has to be downsampled, the
            original sample frequency has to be given as
            an integer. if no integer is given, no
            resampling is performed.
    
    Returns:
        - tap_ind (list of lists): containing one tap per
            list, and per tap a list of the 6 timepoints
            of tapping detected.
        - impacts: indices of impact-moments (means: moment
            of finger-close on thumb)
        - acc_arr (array): preprocessed data array
    """
    if already_preprocd == False:
        # print('preprocessing raw ACC-data')
        if type(orig_fs) == int:
            # print('resample data array')
            acc_arr = resample(acc_arr,
                acc_arr.shape[0] // (orig_fs // fs))

        acc_arr, main_ax_i = run_preproc_acc(
            dat_arr=acc_arr,
            fs=fs,
            to_detrend=True,
            to_check_magnOrder=True,
            to_check_polarity=True,
            verbose=True
        )

    else:
        main_ax_i = find_main_axis(acc_arr, method='variance')
        
    tap_ind, impacts = updrsTapDetector(
        acc_triax=acc_arr, fs=fs, main_ax_i=main_ax_i
    )

    return tap_ind, impacts, acc_arr


