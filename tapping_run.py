'''Run UPDRS FingerTap Detection and Assessment functions'''

# Import public packages and functions
import numpy as np
from itertools import compress
from typing import Any
from array import array
from pandas import DataFrame
from dataclasses import field

# Import own functions
from tap_load_data.tapping_time_detect import updrsTapDetector
from tap_load_data.tapping_preprocess import run_preproc_acc, find_main_axis
from retap_utils.utils_preprocessing import resample


def run_updrs_tap_finder(
    acc_arr: array,
    fs: int,
    already_preprocd: bool=True,
    goal_fs: int = 250,
    main_axis_method: str = 'minmax',
    verbose: bool = False,
):
    """
    Input:
        - acc_arr (array): tri-axial acc array
        - fs (int): sampling freq in Hz - in case of
            resampling: this is the wanted fs where the
            data should be converted to.
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
    # input variable checks
    if type(acc_arr) == DataFrame: acc_arr = acc_arr.values()
    if np.logical_and(
        acc_arr.shape[1] == 3,
        acc_arr.shape[0] > acc_arr.shape[1]
    ): acc_arr = acc_arr.T

    if already_preprocd == False:

        if fs != goal_fs:
            acc_arr = resample(
                data=acc_arr,
                Fs_orig=fs,
                Fs_new=goal_fs
            )
            fs = goal_fs
            if verbose:
                print(
                    f'resample data array: from {fs} Hz to {goal_fs} Hz'
                    f' new shape {acc_arr.shape}'
                )

        acc_arr, main_ax_i = run_preproc_acc(
            dat_arr=acc_arr,
            fs=fs,
            to_detrend=True,
            to_check_magnOrder=True,
            to_check_polarity=True,
            verbose=True
        )

    else:
        main_ax_i = find_main_axis(acc_arr, method=main_axis_method)
        
    tap_ind, impacts = updrsTapDetector(
        acc_triax=acc_arr, fs=fs, main_ax_i=main_ax_i
    )

    return tap_ind, impacts, acc_arr, fs


