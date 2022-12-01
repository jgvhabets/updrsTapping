"""
Utilisation functions for Feature Extraction

part of (updrsTapping-repo)
ReTap-Toolbox
"""

# import public packages and functions
import os
from dataclasses import dataclass, field
from typing import Any
from itertools import product
from pandas import read_csv
from numpy import logical_and, isnan, loadtxt

import tap_extract_fts.tapping_extract_features as ftExtr
import tapping_run as tap_finder
import retap_utils.utils_dataManagement as utils_dataMangm



@dataclass(init=True, repr=True,)
class FeatureSet:
    """
    Class to get meta-data, acc-signals, and
    features for all defined subjecta.

    Returns separate class with all data and info
    available per 10-sec tapping trace

    tapDict are contains lists per tap, containing
    7 detected time points, representing:
    [   startUP, fastestUp, stopUP, startDown, 
        fastestDown, impact, stopDown]
    NOTE: not all 7 timepoints are always present,
    at impact (5) and stopDown (6) are present
    """
    subs_incl: Any = 'ALL'
    centers_incl: list = field(
        default_factory=lambda: ['BER', 'DUS'])
    states: list = field(
        default_factory=lambda: ['M0S0', 'M0S1', 'M1S0', 'M1S1'])
    sides: list = field(
        default_factory=lambda: ['L', 'R'])
    incl_meta_data: bool = True
    skipped_no_meta: list = field(default_factory=list)
    incl_traces: list = field(default_factory=list)
    verbose: bool = False

    def __post_init__(self,):

        for cen in self.centers_incl:

            if self.verbose: print(f'start with {cen}')
            
            datapath = utils_dataMangm.find_onedrive_path(cen)

            if self.subs_incl == 'ALL':
                subs = list(set(
                    [f.split('_')[0] for f in
                     os.listdir(datapath) if f[:3] == cen]
                ))  # finds unique sub-names
            else:
                subs = self.subs_incl
            
            # import participant log data
            if self.incl_meta_data:
                log = utils_dataMangm.get_participantLog(cen)
                meta = True
            else: meta = False
            
            for n_sub, sub in enumerate(subs):
                print(
                    f'extracting sub {sub} ('
                    f'{round((n_sub + 1) / len(subs) * 100)} % of {cen})'
                )

                if meta: sublog = log[[
                    str(s).upper() == sub.upper() for s in log["subID"]
                ]]

                subfiles = list(set(
                    [f for f in os.listdir(datapath)
                     if f[:6].upper() == sub.upper()]
                ))

                for state, side in product(
                    self.states, self.sides
                ):  # loop over all combis of states and sides
                    
                    # get FILES for state
                    combo_files = list(set(
                        [f for f in subfiles if state in f]
                    ))
                    # select on side (DUS is renamed incl side)
                    combo_files = list(set(
                        [f for f in combo_files if f'_{side}_' in f]
                    ))

                    # get META for correct med and stim state
                    if meta:
                        combolog = sublog[logical_and(
                            sublog['medStatus'] == int(state[1]),
                            sublog['stimStatus'] == int(state[3])
                        )].reset_index(drop=True)

                    # get meta for correct side (DUS renamed)
                    if meta:
                        combolog = combolog[
                            [side[0].lower() in s for s in combolog['side']]
                        ].reset_index(drop=True)
                    
                    # no files for given sub-state-side combo
                    if len(combo_files) == 0:
                        if self.verbose: print(f'no FILES found for {state, side}')
                        continue
                    elif len(combolog) == 0:
                        if self.verbose: print(f'no SCORES found for {state, side}')
                        continue
                        
                    # LOOP AND INCLUDE ALL REPETITIONS PER COMBI
                    for n, f in enumerate(combo_files):
                        # find repetition of sub-state-side
                        if f.split('_')[-2][:5] == 'block':
                            rep = int(f.split('_')[-2][-1])
                        else:
                            rep = n + 1

                        # extract updrs tap-score from log-excel
                        if meta:
                            try:
                                tap_score = int(combolog[
                                    combolog['repetition'] == rep
                                ]['updrsFt'])
                            
                            except ValueError:
                                if isnan(
                                    combolog[combolog['repetition'] == rep]['updrsFt']
                                 ).item():
                                    print(
                                        '\tskip extraction - NaN-score '
                                        f'{sub}_{state}_{side}_{rep}'
                                    )
                                    self.skipped_no_meta.append(
                                        f'{sub}_{state}_{side}_{rep}'
                                    )
                                    continue

                            except KeyError:
                                print(
                                        '\tskip extraction - missing-score '
                                        f'{sub}_{state}_{side}_{rep}'
                                    )
                                self.skipped_no_meta.append(
                                    f'{sub}_{state}_{side}_{rep}'
                                )
                                continue
                        
                            except TypeError:
                                if combolog.shape[0] == 0:
                                    print(
                                        '\tskip extraction - missing-score '
                                        f'{sub}_{state}_{side}_{rep}'
                                    )
                                    self.skipped_no_meta.append(
                                        f'{sub}_{state}_{side}_{rep}'
                                    )
                                    continue
                                else:
                                    print(f'REP is {rep}, {sub, state, side}')
                                    print(combolog)
                                    raise TypeError('typeError not because of empty combolog DF')
                        
                        # if meta data is not wanted
                        else:
                            tap_score = None
                        
                        setattr(
                            self,
                            f'{sub}_{state}_{side}_{rep}',
                            singleTrace(
                                sub=sub,
                                state=state,
                                side=side,
                                rep=rep,
                                center=cen,
                                filepath=os.path.join(datapath, f),
                                tap_score=tap_score,
                                to_extract_feats=True,
                            )
                        )

                        self.incl_traces.append(f'{sub}_{state}_{side}_{rep}')


@dataclass(repr=True, init=True,)
class singleTrace:
    """
    Class to store meta-data, acc-signals,
    and features of one single 10-sec tapping trace
    """
    sub: str
    state: str
    side: str
    rep: int
    center: str
    filepath: str
    tap_score: Any
    goal_Fs: int = 250
    to_extract_feats: bool = True

    def __post_init__(self,):
        # load and store tri-axial ACC-signal
        if self.center == 'BER':
            dat = read_csv(self.filepath, index_col=False)
            # delete index col without heading if present
            if 'Unnamed: 0' in dat.keys():
                del(dat['Unnamed: 0'])
                dat.to_csv(self.filepath, index=False)

            dat = dat.values.T  # only np-array as acc-signal
            preproc_bool=True

        elif self.center == 'DUS':  
            # matlab-saved CSV-files
            dat = loadtxt(self.filepath, delimiter='\t')
            preproc_bool=False

        # set data to attribute (3 rows, n-samples columns)
        setattr(self, 'acc_sig', dat)
        
        # extract sample freq if given
        fpart = self.filepath.split('_')[-1]
        fs = fpart.lower().split('hz')[0]
        self.fs = int(fs)
        if self.center == 'DUS': self.fs = 4000

        if self.to_extract_feats:

            tap_idx, impact_idx, new_accsig, new_fs = tap_finder.run_updrs_tap_finder(
                acc_arr=self.acc_sig,
                fs=self.fs,
                goal_fs=self.goal_Fs,
                already_preprocd=preproc_bool,
            )
            setattr(self, 'impact_idx', impact_idx)

            if preproc_bool == False:
                setattr(self, 'acc_sig', new_accsig)
                setattr(self, 'fs', new_fs)
            
            if len(impact_idx) < 10:
                print(
                    f'\tonly {len(impact_idx)} taps for '
                    f'{self.sub}, {self.state}, {self.sub},'
                    f' {self.side}, {self.rep}    '
                    f'(subscore: {self.tap_score})'
                )

            self.fts = ftExtr.tapFeatures(
                triax_arr=self.acc_sig,
                fs=self.fs,
                impacts=self.impact_idx,
                tapDict=tap_idx,
                updrsSubScore=self.tap_score,
            )


### PUT IN SEPERATE PY FILE  
import datetime as dt
# import os
# import retap_utils.utils_dataManagement as utils_dataMangm

if __name__ == '__main__':
    """
    run from main repo path on WINDOWS and macOS as:

        python -m tap_extract_fts.main_featExtractionClass
    
        -m is there because it is ran from a module/package 
        (within a folder)
    """
    data = FeatureSet(
        # subs_incl = ['BER056', ],
        centers_incl = ['BER', 'DUS'],   # 'DUS'
        verbose=False,
    )

    deriv_path = os.path.join(
        utils_dataMangm.get_local_proj_dir(),
        'data',
        'derivatives'
    )
    dd = str(dt.date.today().day).zfill(2)
    mm = str(dt.date.today().month).zfill(2)
    fname = (f'ftClass_ALL_{dt.date.today().year}{mm}{dd}')
    

    utils_dataMangm.save_class_pickle(
        class_to_save=data,
        path=deriv_path,
        filename=fname,
    )


