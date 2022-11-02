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

from retap_utils import utils_dataManagement

@dataclass(init=True, repr=True,)
class FeatureSet:
    """

    """
    subs_incl: Any = 'ALL'
    centers_incl: list = field(
        default_factory=lambda: ['BER', 'DUS'])
    states: list = field(
        default_factory=lambda: ['M0S0', 'M0S1', 'M1S0', 'M1S1'])
    sides: list = field(
        default_factory=lambda: ['L', 'R'])

    def __post_init__(self,):

        for cen in self.centers_incl:
            print(f'start with {cen}')
            datapath = utils_dataManagement.find_stored_data_path(cen)

            if self.subs_incl == 'ALL':
                subs = list(set(
                    [f.split('_')[0] for f in
                     os.listdir(datapath) if f[:3] == cen]
                ))  # finds unique sub-names
            else:
                subs = self.subs_incl
            
            for sub in subs:
                print(f'\tSTART sub: {sub}')
                files = list(set([f for f in os.listdir(datapath) if f[:6] == sub]))

                for combo in product(
                    self.states, self.sides
                ):  # loop over all combis of states and sides

                    combo_files = list(set(
                        [f for f in files if
                            np.logical_and(combo[0] in f, combo[1] in f)]
                    ))
                    # no files for given sub-state-side combo
                    if len(combo_files) == 0: continue

                    for n, f in enumerate(combo_files):
                        if f.split('_')[3][:5] == 'block':
                            rep = int(f.split('_')[3][-1])
                        else:
                            rep = n + 1

                        setattr(
                            self,
                            f'{sub}_{combo[0]}_{combo[1]}_{rep}',
                            singleTrace(
                                sub=sub,
                                state=combo[0],
                                side=combo[1],
                                rep=rep,
                                center=cen,
                                filepath=os.path.join(datapath, f)
                            )
                        )


@dataclass(repr=True, init=True,)
class singleTrace:
    sub: str
    state: str
    side: str
    rep: int
    center: str
    filepath: str

    def __post_init__(self,):
        # store only np-array as acc-signal
        try:
            self.acc_sig = read_csv(self.filepath).values()
        except:
            if os.path.exists(self.filepath):
                print(f'CSV reading Error in {self.filepath}')
            else:
                print(f'{self.filepath} doesnot exist!')
            
            