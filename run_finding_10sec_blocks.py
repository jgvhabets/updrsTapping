"""
Run finding and splitting blocks of 10-seconds
from command line
"""



# import public packages
import os
import sys
from dataclasses import field, dataclass
import numpy as np
from array import array
from typing import Any

# import own functions
from utils import utils_dataManagement, tmsi_poly5reader, utils_preprocessing
import tap_load_data.tapping_find_blocks as find_blocks
import tap_load_data.tapping_preprocess as preproc

 

# find relative path
proj_dir = utils_dataManagement.get_proj_dir()


@dataclass(init=True, repr=True)
class triAxial:
    """
    Select accelerometer keys

    TODO: add Cfg variable to indicate
    user-specific accelerometer-key
    """
    data: array
    key_indices: dict

    def __post_init__(self,):

        try:
            self.left = self.data[
                self.key_indices['L_X']:
                self.key_indices['L_Z'] + 1
            ]
        except KeyError:
            print('No left indices')

        try:
            self.right = self.data[
                self.key_indices['R_X']:
                self.key_indices['R_Z'] + 1
            ]
        except KeyError:
            print('No right indices')


@dataclass(init=True, repr=True)
class rawAccData:
    """
    """
    sub: str
    state: str
    uncut_path: str
    joker_string: Any = None
    goal_fs: int = 250

    def __post_init__(self,):
        
        sel_files = utils_dataManagement.get_file_selection(
            path=self.uncut_path,
            sub=self.sub, state=self.state,
            joker_string=self.joker_string
        )
        print(f'files selected: {sel_files}')

        for f in sel_files:

            self.raw = tmsi_poly5reader.Poly5Reader(
                os.path.join(self.uncut_path, f)
            )
            key_ind_dict = utils_dataManagement.get_arr_key_indices(
                self.raw.ch_names
            )
            if len(key_ind_dict) == 0:
                print(f'No ACC-keys found in keys: {self.raw.ch_names}')
                continue
            
            print(key_ind_dict)

            # select only present acc (aux) variables
            temp_data = triAxial(
                data=self.raw.samples,
                key_indices=key_ind_dict,
            )

            # resampling before acc-selection was here

            # self.data = triAxial(
            #     data=temp_data,
            #     key_indices=key_ind_dict,
            # )

            for acc_side in vars(temp_data).keys():

                if acc_side in ['left', 'right']:

                    if self.raw.sample_rate > self.goal_fs:

                        setattr(
                            temp_data,
                            acc_side,
                            utils_preprocessing.resample(
                                getattr(temp_data, acc_side),
                                Fs_orig=self.raw.sample_rate,
                                Fs_new=self.goal_fs
                            )
                        )

                    setattr(
                        temp_data,
                        acc_side,
                        preproc.run_preproc_acc(
                            getattr(temp_data, acc_side),
                            fs=self.goal_fs,
                            to_detrend=True,
                            to_check_magnOrder=True,
                            to_check_polarity=False,
                        )
                    )

                    # main_ax_i = preproc.find_main_axis(
                    #     getattr(temp_data, acc_side)
                    # )

                    # setattr(
                    #     temp_data,
                    #     acc_side,
                    #     preproc.check_order_magnitude(
                    #         getattr(temp_data, acc_side),
                    #         main_ax_i,
                    #     )
                    # )

                    # setattr(
                    #     temp_data,
                    #     acc_side,
                    #     preproc.detrend_bandpass(
                    #         getattr(temp_data, acc_side),
                    #         fs=self.goal_fs,
                    #     )
                    # )

                    self.data = temp_data

                    temp_acc, temp_ind = find_blocks.find_active_blocks(
                        acc_arr=getattr(temp_data, acc_side),
                        fs=self.goal_fs,
                        verbose=True,
                        to_plot=True,
                        figsave_dir=os.path.join(
                            proj_dir, 'figures', 'testRepo'
                        ),
                        figsave_name=(
                            f'{self.sub}_{self.state}'
                            f'{acc_side}_blocks_detected'
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








