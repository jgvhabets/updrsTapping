"""
Utilisation functions to support
preprocessing steps in ReTap-Toolbox
"""

# import public packages
from array import array

import numpy as np
from scipy.signal import resample_poly

def resample(
    data: array,
    Fs_orig: int,
    Fs_new: int
):
    """
    Downsampling of recorded (acc) data to the
    desired frequency for feature extraction.
    Function is written for downsampling since
    this will always be the case.

    Arguments:
        - data (array): 2d or 3d array with data,
        first dimension are windows, second dim are
        rows, third dim are the data points over time
        within one window
        - Fs_orig (int): original sampling freq
        - Fs_new (int): desired sampling freq

    Returns:
        - newdata (array): containing similar
        data arrays as input data, however downsampled
        and therefore less datapoints per window
    """
    down = int(Fs_orig / Fs_new)  # factor to down sample

    newdata = resample_poly(
        data, up=1, down=down, axis=-1
    )

    return newdata