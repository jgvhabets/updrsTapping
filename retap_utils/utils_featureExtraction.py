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
from pandas import read_csv, read_excel
from numpy import logical_and

from retap_utils import utils_dataManagement
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
    verbose: bool = False

    def __post_init__(self,):

        for cen in self.centers_incl:
            if self.verbose: print(f'start with {cen}')
            datapath = utils_dataManagement.find_stored_data_path(cen)

            if self.subs_incl == 'ALL':
                subs = list(set(
                    [f.split('_')[0] for f in
                     os.listdir(datapath) if f[:3] == cen]
                ))  # finds unique sub-names
            else:
                subs = self.subs_incl
            
            # import participant log data
            if self.incl_meta_data:
                log = get_participantLog(cen)
                meta = True
            else: meta = False
            
            for sub in subs:
                if self.verbose: print(f'\tSTART sub: {sub}')

                if meta: sublog = log[[
                    str(s).upper() == sub.upper() for s in log["subID"]
                ]]

                subfiles = list(set(
                    [f for f in os.listdir(datapath)
                     if f[:6].upper() == sub.upper()]
                ))

                for combo in product(  # TODO: splits combo in state and side
                    self.states, self.sides
                ):  # loop over all combis of states and sides
                    
                    # get files for state and side
                    if cen == 'BER':
                        combo_files = list(set(
                            [f for f in subfiles if
                            logical_and(combo[0] in f, combo[1] in f)]
                        ))
                    elif cen == 'DUS':  # DUS dta is without sides
                        combo_files = list(set(
                            [f for f in subfiles if combo[0] in f]
                        ))

                    # get meta for correct med and stim state
                    if meta: combolog = sublog[logical_and(
                        sublog['medStatus'] == int(combo[0][1]),
                        sublog['stimStatus'] == int(combo[0][3])
                    )].reset_index(drop=True)

                    if cen == 'BER':  # get meta for correct side
                        if meta: combolog = combolog[
                            [combo[1][0].lower() in s for s in combolog['side']]
                        ].reset_index(drop=True)
                    
                    # no files for given sub-state-side combo
                    if len(combo_files) == 0:
                        if self.verbose: print(f'no files found for {combo}')
                        continue

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
                                print('tapscore', tap_score)
                                
                            except KeyError:
                                print('meta not available in log for '
                                      f'{sub}_{combo[0]}_{combo[1]}_{rep}')
                                continue
                            except TypeError:
                                print(rep,)
                                print(combo)
                                print(combo_files)
                                raise TypeError(f'rep is {rep}')
                                continue
                        else:
                            tap_score = None

                        setattr(
                            self,
                            f'{sub}_{combo[0]}_{combo[1]}_{rep}',
                            singleTrace(
                                sub=sub,
                                state=combo[0],
                                side=combo[1],
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
        # store only np-array as acc-signal
        dat = read_csv(self.filepath, index_col=False)
        # delete index col without heading if present
        if 'Unnamed: 0' in dat.keys():
            del(dat['Unnamed: 0'])
            dat.to_csv(self.filepath, index=False)
        # set data to attribute
        setattr(self, 'acc_sig', dat.values.T)

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
            print(f'# {len(impact_idx)} tap found')

            fts = ftExtr.tapFeatures(
                triax_arr=self.acc_sig,
                fs=self.fs,
                impacts=impact_idx,
                tapDict=tap_idx,
                updrsSubScore=self.tap_score,
            )
            print(f'{self.sub, self.state, self.side, self.tap_score}'
                '  features extracted\n')
            setattr(self, 'fts', fts)




        

def get_participantLog(center = ['DUS', 'BER']):
    """
    Get Excel file with participant
    meta data "ReTap_participantLog.xlsx"

    Input:
        - center: optional, DUS or BER, defaults
            to both. Defines which excel-sheets
            are imported

    Returns:
        - log: dict containing the BER and DUS
            sheet, each in one DataFRame in dict
    """
    p = utils_dataManagement.find_stored_data_path('retapdata')
    xl_fname = 'ReTap_participantLog.xlsx'

    log = read_excel(
        os.path.join(p, xl_fname),
        sheet_name=center
    )

    return log
