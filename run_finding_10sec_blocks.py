"""
Run finding and splitting blocks of 10-seconds
from command line
"""



# import public packages
import os
import sys
from dataclasses import dataclass
import numpy as np
from array import array

# import own functions
from utils import utils_dataManagement, tmsi_poly5reader, utils_preprocessing
import tap_load_data.tapping_find_blocks as find_blocks
import tap_load_data.tapping_preprocess as preproc

 

# find relative path
proj_dir = utils_dataManagement.get_proj_dir()


@dataclass(init=True, repr=True)
class triAxial:
    data: array
    key_indices: dict

    def __post_init__(self,):

        self.left = self.data[
            self.key_indices['L_X']:
            self.key_indices['L_Z'] + 1
        ]
        self.right = self.data[
            self.key_indices['R_X']:
            self.key_indices['R_Z'] + 1
        ]


@dataclass(init=True, repr=True)
class rawAccData:
    """
    """
    sub: str
    state: str
    uncut_path: str
    goal_fs: int = 250

    def __post_init__(self,):

        for f in os.listdir(self.uncut_path):

            if self.sub not in f[:10]: continue

            if self.state not in f: continue

        raw = tmsi_poly5reader.Poly5Reader(
            os.path.join(self.uncut_path, f)
        )
        key_ind_dict = utils_dataManagement.get_arr_key_indices(
            raw.ch_names
        )
        print(key_ind_dict)

        if raw.sample_rate > self.goal_fs:

            print(f'old shape {raw.samples.shape}')

            resampled = utils_preprocessing.resample(
                raw.samples,
                Fs_orig=raw.sample_rate,
                Fs_new=self.goal_fs
            )

            self.data = triAxial(
                data=resampled,
                key_indices=key_ind_dict,
            )

            for side in ['left', 'right']:

                main_ax_i = preproc.find_main_axis(
                    getattr(self.data, side)
                )

                setattr(
                    self.data,
                    side,
                    preproc.check_order_magnitude(
                        getattr(self.data, side),
                        main_ax_i,
                    )
                )

                setattr(
                    self.data,
                    side,
                    preproc.detrend_bandpass(
                        getattr(self.data, side),
                        fs=self.goal_fs,
                    )
                )

                temp_acc, temp_ind = find_blocks.find_active_blocks(
                    acc_arr=getattr(self.data, side),
                    fs=self.goal_fs,
                    verbose=True,
                    to_plot=True,
                    figsave_dir=os.path.join(
                        proj_dir, 'figures', 'testRepo'
                    ),
                    figsave_name=(
                        f'{self.sub}_{self.state}'
                        f'{side}_blocks_detected'
                    )
                )




    

if __name__ == '__main__':
    # only executes following code when called via
    # command line, not when loaded in, in another
    # script
    raw = rawAccData(
        sub=sys.argv[1],
        state=sys.argv[2],
        uncut_path=sys.argv[3],
    )








