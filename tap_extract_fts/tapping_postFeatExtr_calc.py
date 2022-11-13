"""
Feature calculations functions
"""

# Import public packages and functions
import numpy as np
from scipy.stats import linregress, variation



def ft_decrement(
    ft_array: list,
    method: str,
    n_taps_mean: int = 5,
):
    """
    Calculates the proportional decrement within
    feature values per tap.
    Positive decrement means increase in feature,
    negative decrement means decrease over time.

    If less than 10 tap-values available, np.nan
    is returned

    Inputs:
        - ft_array: feature values (one per tap), to
            calculate the decrement over
        - method: method to calculate decrement:
            - diff_in_mean calculates the normalised
                difference between first and last taps
            - regr_slope takes the normalised slope
                of a fitted linear regression line
        - n_taps_mean: numer of taps taking to average
            beginning and end taps (only in means method)
    """
    avail_methods = ['diff_in_mean', 'regr_slope']
    assert method in avail_methods, ('method for '
        f'decrement calc should be in {avail_methods}'
    )

    if len(ft_array) < 10:
            
        return np.nan

    # loop over arrays with amp-values
    if method == 'diff_in_mean':

        startMean = np.nanmean(ft_array[:n_taps_mean])
        endMean = np.nanmean(ft_array[-n_taps_mean:])

        if np.isnan(startMean): return np.nan

        # decrement is difference between end and start
        # normalised against 90-perc of max amplitude
        decr = (endMean - startMean) / startMean

        return decr

    elif method == 'regr_slope':
        ft_array = ft_array[~np.isnan(ft_array)]  # exclude nans
        slope, intercept = np.polyfit(
            np.arange(len(ft_array)), ft_array, 1)

        return slope



def aggregate_arr_fts(
    method, ft_array
):
    """
    Aggregate array-features (calculated
    per tap in block) to one value per
    block.
    """
    assert method in [
        'mean', 'median', 'stddev', 'sum', 'variance',
        'coefVar', 'trend_slope', 'trend_R'
    ], f'Inserted method "{method}" is incorrect'

    if np.isnan(ft_array).any():

        ft_array = ft_array[~np.isnan(ft_array)]

    if ft_array.size == 0:
        print('artificial 0 added')
        return np.nan

    if method == 'allin1':

        if np.isnan(ft_array).any():
            ft_array = ft_array[~np.isnan(ft_array)]

        return ft_array  # all in one big list

    elif method == 'mean':
        
        return np.nanmean(ft_array)
    
    elif method == 'median':
        
        return np.nanmedian(ft_array)

    elif method == 'stddev':

        ft_array = normalize_var_fts(ft_array)
        
        return np.nanstd(ft_array)

    elif method == 'sum':
        
        return np.nansum(ft_array)

    elif method == 'coefVar':

        # ft_array = normalize_var_fts(ft_array)
        cfVar = np.nanstd(ft_array) / np.nanmean(ft_array)
        # taking nan's into account instead of variation()

        return cfVar
    
    elif method == 'variance':

        ft_array = normalize_var_fts(ft_array)

        return np.var(ft_array)

    elif method[:5] == 'trend':

        try:
            linreg = linregress(
                np.arange(ft_array.shape[0]),
                ft_array
            )
            slope, R = linreg[0], linreg[2]

            if np.isnan(slope):
                slope = 0

            if method == 'trend_slope': return slope
            if method == 'trend_R': return R

        except ValueError:
            
            return 0


def normalize_var_fts(values):

    ft_max = np.nanmax(values)
    ft_out = values / ft_max

    return ft_out


def nan_array(dim: list):
    """Create 2 or 3d np array with nan's"""
    if len(dim) == 2:
        arr = np.array(
            [[np.nan] * dim[1]] * dim[0]
        )
    else:
        arr = np.array(
            [[[np.nan] * dim[2]] * dim[1]] * dim[0]
        ) 

    return arr



# ### SMOOTHING FUNCTION WITH NP.CONVOLVE

# sig = accDat['40'].On
# dfsig = np.diff(sig)

# kernel_size = 10
# kernel = np.ones(kernel_size) / kernel_size
# sigSm = np.convolve(sig, kernel, mode='same')
# dfSm = np.convolve(dfsig, kernel, mode='same')

# count = 0
# for i, df in enumerate(dfSm[1:]):
#     if df * dfSm[i] < 0: count += 1

# plt.plot(sigSm)
# plt.plot(dfSm)

# plt.xlim(1000, 1500)


# print(count)