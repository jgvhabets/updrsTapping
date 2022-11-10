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

from retap_utils import utils_dataManagement as utils_dataMangm
from tap_extract_fts import tapping_extract_features as ftExtr
import tapping_run as tap_finder

@dataclass(init=True, repr=True,)
class FeatureSet:
    """
    Class to get meta-data, acc-signals, and
    features for all defined subjecta.

    Returns separate class with all data and info
    available per 10-sec tapping trace
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
                    f'\textracting sub {sub} ('
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

        elif self.center == 'DUS':  # matlab-saved CSV-files
            dat = loadtxt(self.filepath, delimiter='\t')

        # set data to attribute (3 rows, n-samples columns)
        setattr(self, 'acc_sig', dat)

        # extract sample freq if given
        if 'hz' in self.filepath.lower():
            fpart = self.filepath.split('_')[-1]
            fs = fpart.lower().split('hz')[0]
            self.fs = int(fs)
        else:
            self.fs = 250  # defaults to 250 if not defined in filename

        if self.to_extract_feats:

            tap_idx, impact_idx, _ = tap_finder.run_updrs_tap_finder(
                acc_arr=self.acc_sig,
                fs=self.fs,
                already_preprocd=True,
            )
            setattr(self, 'impact_idx', impact_idx)

            self.fts = ftExtr.tapFeatures(
                triax_arr=self.acc_sig,
                fs=self.fs,
                impacts=self.impact_idx,
                tapDict=tap_idx,
                updrsSubScore=self.tap_score,
            )

        

