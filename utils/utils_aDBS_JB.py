"""
Utilisation functions for aDBS-JB analyses
"""

from dataclasses import dataclass
from typing import Any

@dataclass(init=True, repr=True,)
class aDBS_subjects:
    """
    """
    meta: Any
    ftsAll: Any

    def __post_init__(self,):

        self.subs = self.meta['subject'].unique()
        run_ids = list(self.ftsAll.keys())

        for sub in self.subs:

            sub_inds = list(
                self.meta[self.meta['subject'] == sub].index
            )

            setattr(
                self,
                f'sub{sub}',
                aDBS_subRuns(
                    sub=sub,
                    sub_run_ids=sub_inds,
                    meta=self.meta,
                    fts=self.ftsAll
                )
            )            


@dataclass(init=True, repr=True,)
class aDBS_subRuns:
    """
    Create Class per subject with fts
    from all stim-settings
    """
    sub: str
    sub_run_ids: list
    meta: Any
    fts: Any

    def __post_init__(self,):

        list_runnames = []

        for i in self.sub_run_ids:

            assert self.meta.iloc[i]["subject"] == self.sub, print(
                f'Sub in Index-{i} doesnt match with Sub {self.sub}'
            )

            if self.meta.iloc[i]['stim'] == 'cDbsOff':
                run_name = 'cOff'  # excl from runname sub{sub}_

            elif self.meta.iloc[i]['stim'] == 'cDbsOn':
                run_name = 'cOn'
                
            elif self.meta.iloc[i]['stim'][0] == 'a':
                run_name = (
                    f'a{self.meta.iloc[i]["stim"][-1]}_'
                    f'{self.meta.iloc[i]["trial"]}'
                )

            fts_run_name = list(self.fts.keys())[i]
            
            setattr(
                self,
                run_name,
                aDBS_run(
                    sub=self.sub,
                    run_name=run_name,
                    stim=self.meta.iloc[i]['stim'],
                    trial=self.meta.iloc[i]['trial'],
                    fts=self.fts[fts_run_name]
                )
            )

            list_runnames.append(run_name)
        
        self.run_names = list_runnames


@dataclass(init=True, repr=True,)
class aDBS_run:
    """
    """
    sub: str
    run_name: str  # was run_id
    stim: str
    trial: str
    fts: Any
