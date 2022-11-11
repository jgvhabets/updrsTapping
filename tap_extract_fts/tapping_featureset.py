'''
Functions to define tapping features
'''
# Import general packages and functions
import numpy as np

# Import own functions
# from tap_load_data import tapping_preprocess


def signalvectormagn(acc_arr):
    """
    Input:
        - acc_arr (array): triaxial array
            with x-y-z axes (3 x n_samples)
    
    Returns:
        - svm (array): uniaxial array wih
            signal vector magn (1, n_samples)
    """
    if acc_arr.shape[0] != 3: acc_arr = acc_arr.T
    assert acc_arr.shape[0] == 3, ('Array must'
    'be tri-axial (x, y, z from accelerometer)')
  
    svm = np.sqrt(
        acc_arr[0] ** 2 +
        acc_arr[1] ** 2 +
        acc_arr[2] ** 2
    )

    return svm


def calc_RMS(acc_signal):
    """
    Calculate RMS over preselected uniaxial
    acc-signal (can be uni-axis or svm)
    """
    S = np.square(acc_signal)
    MS = S.mean()
    RMS = np.sqrt(MS)

    return RMS


def tap_RMS(
    tapDict,
    triax_arr,
    ax,
    to_norm: bool = False,
    select: str='tap',
    impact_window: float=.25,
    fs: int = None
):
    """
    Calculates RMS of full acc-signal per tap.

    Input:
        - tapDict: dict with list of indices
            representing tap [startUP, fastestUp,
            stopUP, startDown, fastestDown, impact,
            stopDown] (result of updrsTapDetector())
        - triax_arr (array)
        - ax: index of main-axis
        - select (str): if full -> return RMS per
            full tap; if impact -> return RMS
            around impact
        - to_norm: bool defining to normalise RMS
            to duration of originating timeframe
        - impact_window (float): in seconds, total
            window around impact to calculate RMS
        - fs (int): sample frequency, required for
            impact-RMS and normalisation
    
    Returns:
        - RMS_uniax (arr)
        - RMS_triax (arr)
    """
    select_options = ['run', 'tap', 'impact']
    assert select in select_options, ('select '
        'select variable incorrect for tap_RMS()')
    if np.logical_or(to_norm, select == 'impact'):
        assert type(fs) == int, print(
            'Frequency has to be given as integer'
        )

    ax = triax_arr[ax]
    svm = signalvectormagn(triax_arr)

    if select == 'run':
        RMS_uniax = calc_RMS(ax)
        if to_norm: RMS_uniax /= (len(ax) / fs)  # divide RMS by length in sec
        
        RMS_triax = calc_RMS(svm)
        if to_norm: RMS_triax /= (len(svm) / fs)

        return RMS_uniax, RMS_triax
    
    else:
        RMS_uniax = np.zeros(len(tapDict))
        RMS_triax = np.zeros(len(tapDict))

        for n, tap in enumerate(tapDict):
            tap = tap.astype(int)  # np.nan as int is -999999...

            if select == 'tap':
                sel1 = int(tap[0])
                sel2 = int(tap[-1])
                if np.isnan(sel2): sel2 = int(tap[-2])

            elif select == 'impact':
                sel1 = int(tap[-2] - int(fs * impact_window / 2))
                sel2 = int(tap[-2] + int(fs * impact_window / 2))

            if np.logical_or(sel1 == np.nan, sel2 == np.nan):
                print('tap skipped, missing indices')
                continue
            
            tap_ax = ax[sel1:sel2]
            tap_svm = svm[sel1:sel2]
            
            RMS_uniax[n] = calc_RMS(tap_ax)
            RMS_triax[n] = calc_RMS(tap_svm)
        
            if to_norm: RMS_uniax[n] /= (len(tap_ax) / fs)
            if to_norm: RMS_triax[n] /= (len(tap_svm) / fs)
        
        return RMS_uniax, RMS_triax


def upTap_velocity(tapDict, triax_arr, ax):
    """
    Calculates velocity approximation via
    area under the curve of acc-signal within
    upwards part of a tap.

    Input:
        - tapDict
        - triax_arr
        - ax (int): main tap axis index
    
    Returns:
        - upVelo_uniax (arr)
        - upVelo_triax (arr)
    """
    ax = triax_arr[ax]
    svm = signalvectormagn(triax_arr)

    upVelo_uniax = velo_AUC_calc(tapDict, ax)
    upVelo_triax = velo_AUC_calc(tapDict, svm)
    
    return upVelo_uniax, upVelo_triax


def intraTapInterval(
    tapDict: dict,
    fs: int,
    moment: str = 'impact'
):
    """
    Calculates intratap interval.

    Input:
        - tapDict: dict with lists resulting
            [startUP, fastestUp, stopUP, startDown,
            fastestDown, impact, stopDown]
            (result of updrsTapDetector())
        - fs (int): sample freq
        - moment: timing of ITI calculation, from
            impact-imapct, or start-start
    
    Returns:
        - iti (arr): array with intratapintervals
            in seconds
    """
    assert moment.lower() in ['impact', 'start'], print(
        f'moment ({moment}) should be start or impact'
    )
    if moment.lower() == 'start': idx = 0
    elif moment.lower() == 'impact': idx = -2
    
    iti = np.array([np.nan] * (len(tapDict) - 1))

    for n in np.arange(len(tapDict) - 1):
        # take distance between two impact-indices
        distance = tapDict[n + 1][idx] - tapDict[n][idx]
    
        iti[n] = distance / fs  # from samples to seconds
    
    return iti

# import matplotlib.pyplot as plt


def velo_AUC_calc(tapDict, accSig,):
    """
    Calculates max velocity during finger-raising
    based on the AUC from the first big pos peak
    in one tap until the acceleration drops below 0

    Input:
        - tapDict: dict with lists resulting
            [startUP, fastestUp, stopUP, startDown,
            fastestDown, impact, stopDown]
            (result of updrsTapDetector())
        - accSig (array): uniax acc-array (one ax or svm)
    
    Returns:
        - out (array): one value or nan per tap in tapDict
    """
    out = []

    for n, tap in enumerate(tapDict):

        if ~np.isnan(tap[1]):  # crossing 0 has to be known
            # take acc-signal [start : fastest point] of rise
            line = accSig[int(tap[0]):int(tap[1])]
            areas = []
            for s, y in enumerate(line[1:]):
                areas.append(np.mean([y, line[s]]))
            if sum(areas) == 0:
                print('\nSUM 0',n, line[:30], tap[0], tap[1])
            out.append(sum(areas))
    
    return np.array(out)


def amplitudeDecrement(
    tapAmpFts: list, width_sel: float=.25,
    min_n_taps: int=10
):
    """
    Sums the proportional decrement of all
    amplitude features calculated per single-
    tap

    TODO: repeat measure with slope of linreg line
    TODO: normalise decrement to max value within trace

    Inputs:
        - list with single-tap-amp features
        - width_sel (float): part of taps at
            beginning and end to determine
            decrement, default=.25
        - min_n_taps (int): minimum amount of
            taps present to calculate decrement
    """
    totalDecr = 0

    # loop over arrays with amp-values
    for ft in tapAmpFts:

        n_taps = len(ft)

        if n_taps < min_n_taps:
            continue

        sel_n = int(width_sel * n_taps)

        startMean = np.nanmean(ft[:sel_n])
        endMean = np.nanmean(ft[-sel_n:])

        # decerement is difference between end and start
        decr = endMean - startMean

        # normalise against overall mean
        decr = decr / np.nanmean(ft)

        # sum up to total
        totalDecr += decr
    
    if totalDecr == 0: totalDecr = -10

    return totalDecr


def smallSlopeChanges(
    accsig, resolution: str, n_hop: int=1,
    tapDict = [], smoothing,
):
    """
    Detects the number of small changes in
    direction of acceleration.
    Hypothesized is that best tappers, have
    the smoothest acceleration-trace and
    therefore lower numbers of small
    slope changes

    TODO: test with smoothing

    Inputs:
        - acc (array): tri-axial acceleration
            signal from e.g. 10-s tapping
        - n_hop (int): the number of samples used
            to determine the difference between
            two points
    
    Returns:
        - count (int): number of times the
            differential of all thee seperate
            axes changed in direction.
    """
    if resolution == 'run':

        count = 0
        for ax in [0, 1, 2]:

            axdiff = np.diff(accsig[ax])
            for i in np.arange(axdiff.shape[0] - n_hop):
                if (axdiff[i + n_hop] * axdiff[i]) < 0:  # removed if -1 < axdiff...
                    count += 1

    elif resolution == 'taps':

        countlist = []

        for tap in tapDict:

            if np.logical_or(
                np.isnan(tap[0]),
                np.isnan(tap[-1])
            ):
                continue

            elif len(tap) == 0:
                continue
            
            else:
                tap_acc = accsig[:, int(tap[0]):int(tap[-1])]
                count = 0

                for ax in [0, 1, 2]:
                    axdiff = np.diff(tap_acc[ax])

                    for i in np.arange(axdiff.shape[0] - n_hop):
                        if (axdiff[i + n_hop] * axdiff[i]) < 0:  # removed if -1 < axdiff...
                            count += 1
                
                countlist.append(count)

        count = np.array(countlist)

        duration_sig = accsig.shape[1] / fs
        norm_count = count / duration_sig

    return norm_count



# ### Plot AUC-Method for velocity
# start=accTaps['40']['On'][6][0]
# stop=accTaps['40']['On'][6][1]
# plt.plot(accDat['40'].On[start:stop], label='Accelerating phase of finger raising')

# line = accDat['40'].On[start:stop]
# areas = []
# for n, y in enumerate(line[1:]):
#     areas.append(np.mean([y, line[n]]))
# plt.bar(
#     np.arange(.5, len(areas) + .5, 1), height=areas,
#     color='b', alpha=.2,
#     label='AUC extracted',
# )
# plt.ylabel('Acceleration (m/s/s)')
# plt.xlabel('Samples (250 Hz)')
# plt.legend(frameon=False)
# plt.savefig(
#     os.path.join(temp_save, 'ACC', 'fingerTap_AUC_method'),
#     dpi=150, facecolor='w',)
# plt.show()

## DEFINE FEATURE FUNCTIONS FROM MAHADEVAN 2020

def histogram(signal_x):
    '''
    Calculate histogram of sensor signal.
    :param signal_x: 1-D numpy array of sensor signal
    :return: Histogram bin values, descriptor
    '''
    descriptor = np.zeros(3)

    ncell = np.ceil(np.sqrt(len(signal_x)))

    max_val = np.nanmax(signal_x.values)
    min_val = np.nanmin(signal_x.values)

    delta = (max_val - min_val) / (len(signal_x) - 1)

    descriptor[0] = min_val - delta / 2
    descriptor[1] = max_val + delta / 2
    descriptor[2] = ncell

    h = np.histogram(signal_x, ncell.astype(int), range=(min_val, max_val))

    return h[0], descriptor


def signal_entropy(winDat):
    data_norm = winDat/np.std(winDat)
    h, d = histogram(data_norm)
    lowerbound = d[0]
    upperbound = d[1]
    ncell = int(d[2])

    estimate = 0
    sigma = 0
    count = 0

    for n in range(ncell):
        if h[n] != 0:
            logf = np.log(h[n])
        else:
            logf = 0
        count = count + h[n]
        estimate = estimate - h[n] * logf
        sigma = sigma + h[n] * logf ** 2

    nbias = -(float(ncell) - 1) / (2 * count)

    estimate = estimate / count
    estimate = estimate + np.log(count) + np.log((upperbound - lowerbound) / ncell) - nbias
    
    # Scale the entropy estimate to stretch the range
    estimate = np.exp(estimate ** 2) - np.exp(0) - 1
    
    return estimate


def jerkiness(winDat, fs):
    """
    jerk ratio/smoothness according to Mahadevan, npj Park Dis 2018
    uses rate of acc-changes (Hogan 2009). PM was aimed for 3-sec windows
    -> double check function with references
    """
    ampl = np.max(np.abs(winDat))
    jerk = winDat.diff(1) * fs
    jerkSqSum = np.sum(jerk ** 2)
    scale = 360 * ampl ** 2 / len(winDat) / fs
    meanSqJerk = jerkSqSum / fs / (len(winDat) / fs * 2)
    jerkRatio = meanSqJerk / scale
    
    return jerkRatio

