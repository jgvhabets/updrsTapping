'''
Functions to define spectral baselines for neuro-
physiology data (LFP and ECOG) in ReTune's Dyskinesia Project

Based on the run that is analyzed, a class is createsd
with baseline raw signals, PSDs (Welch) and wavelet
decomposition based on predefined minutes from the Rest
recording of the corresponding Session. Minutes from Rest
recording are selected to contain no movement.
'''
# Import general packages and functions
from typing import Any
import numpy as np
from dataclasses import dataclass, field

# Import own custom functions
import tap_extract_fts.tapping_featureset as tap_feats
import tap_extract_fts.tapping_postFeatExtr_calc as postExtrCalc
from tap_load_data.tapping_preprocess import find_main_axis

@dataclass(init=True, repr=True, )
class tapFeatures:
    """
    Extract features from detected taps in acc-tapping trace

    Input:
        - triax_arr: 2d-array with tri-axial acc-signal
        - fs (int): sample freq in Hz
        - impacts (array): array containing indices of
            impact (closing finger) moments
        - tap_lists: list of taps, every tap has an own array
            with 7 timestamps (in n samples) representing the
            moments during a tap (resulting from continuous
            tapping detect function)
        - max_n_taps_incl: integer defining the number of taps
            consider during ft extraction, defaults to zero,
            and is not considered when being zero.
        - updrsSubScore: UPDRS III Fingertapping subscore
            corresponding to acc signal, default False
    """
    triax_arr: Any
    fs: int
    impacts: Any
    tap_lists: dict = field(default_factory=dict)
    max_n_taps_incl: int = 0
    updrsSubScore: Any = False
    
    def __post_init__(self,):

        if len(self.tap_lists) == 0:  # no taps detected
            return

        ax = find_main_axis(self.triax_arr, method='minmax',)

        self.total_nTaps = len(self.impacts)
        
        self.freq = self.total_nTaps / (
            self.triax_arr.shape[1] / self.fs)
        
        self.tap_durations = np.diff(self.impacts) / self.fs

        if self.max_n_taps_incl > 0:
            setattr(self, 'tap_lists', self.tap_lists[:self.max_n_taps_incl])
        
        self.intraTapInt = tap_feats.intraTapInterval(
            self.tap_lists, self.fs
        )

        self.tapRMS = tap_feats.RMS_extraction(
            self.tap_lists,
            self.triax_arr,
            acc_select='svm',
            unit_to_assess='taps',
            ax=ax,
        )

        self.tapRMSnrm = tap_feats.RMS_extraction(
            self.tap_lists,
            self.triax_arr,
            acc_select='svm',
            unit_to_assess='taps',
            ax=ax,
            to_norm=True,
            fs=self.fs,
        )

        self.impactRMS = tap_feats.RMS_extraction(
            self.tap_lists,
            self.triax_arr,
            acc_select='svm',
            unit_to_assess='impacts',
            ax=ax,
            fs=self.fs,
        )

        self.raise_velocity = tap_feats.velocity_raising(
            self.tap_lists, self.triax_arr, ax=ax,
        )  # currently only velocity raising based on svm
        
        self.jerkiness_taps = tap_feats.jerkiness(
            accsig=self.triax_arr,
            fs=self.fs,
            tap_indices=self.tap_lists,
            unit_to_assess='taps',
            smooth_samples=0,
        )

        self.jerkiness_trace = tap_feats.jerkiness(
            accsig=self.triax_arr,
            fs=self.fs,
            tap_indices=self.tap_lists,
            unit_to_assess='trace',
            smooth_samples=0,
        )

        if type(self.updrsSubScore) == str or np.str_:
            self.updrsSubScore = float(self.updrsSubScore)

        ### POST-EXTRACTION ANALYSIS
        fts_to_postExtr_calc = [
            'tapRMS',
            'tapRMSnrm',
            'impactRMS',
            'raise_velocity',
            'intraTapInt',
            'jerkiness_taps',
        ]

        for ft in fts_to_postExtr_calc:

            setattr(
                self,
                f'mean_{ft}',
                postExtrCalc.aggregate_arr_fts(
                    ft_array=getattr(self, ft),
                    method='mean',
                )
            )
            setattr(
                self,
                f'coefVar_{ft}',
                postExtrCalc.aggregate_arr_fts(
                    ft_array=getattr(self, ft),
                    method='coefVar',
                )
            )
            setattr(
                self,
                f'IQR_{ft}',
                postExtrCalc.aggregate_arr_fts(
                    ft_array=getattr(self, ft),
                    method='IQR',
                )
            )
       
            setattr(
                self,
                f'decr_{ft}',
                postExtrCalc.ft_decrement(
                    ft_array=getattr(self, ft),
                    method='diff_in_mean',
                    n_taps_mean=3,
                )
            )
            setattr(
                self,
                f'slope_{ft}',
                postExtrCalc.ft_decrement(
                    ft_array=getattr(self, ft),
                    method='regr_slope',
                )
            )

        # clear up space
        self.triax_arr = 'cleaned up'
        
