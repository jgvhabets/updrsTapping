"""
Run finding and splitting blocks of 10-seconds
from command line
"""



# import public packages
import json
import os
import sys
from dataclasses import field, dataclass
import numpy as np
from array import array
from typing import Any

# import own functions
from retap_utils import utils_dataManagement, tmsi_poly5reader, utils_preprocessing
import tap_load_data.tapping_find_blocks as find_blocks
import tap_load_data.tapping_preprocess as preproc

 

# find relative path
proj_dir = utils_dataManagement.get_proj_dir()


@dataclass(init=True, repr=True)
class rawAccData:
    """
    """
    sub: str
    state: str
    uncut_path: str
    joker_string: Any = None
    goal_fs: int = 250
    sub_csv_code: str = 'BER'
    switched_sides: Any = None
    unilateral_coding_list: list = field(
        default_factory=lambda: [
            'LHAND', 'RHAND',
            'FTL', 'FTR',
            'LFTAP', 'RFTAP',
            
        ]
    )

    def __post_init__(self,):
        # IDENTIFY FILES TO PROCESS
        sel_files = utils_dataManagement.get_file_selection(
            path=self.uncut_path,
            sub=self.sub, state=self.state,
            joker_string=self.joker_string
        )
        print(f'files selected: {sel_files}')
        
        # Abort if not file found
        if len(sel_files) == 0:
            return print(f'No files found for {self.sub} {self.state}')

        for f in sel_files:
            # LOAD FILE
            self.raw = tmsi_poly5reader.Poly5Reader(
                os.path.join(self.uncut_path, f)
            )
            hand_code = 'bilat'
            # check if file contains unilateral data
            for code in self.unilateral_coding_list:
                if code.upper() in f.upper():
                    hand_code = code.upper()

            key_ind_dict, file_side = utils_dataManagement.get_arr_key_indices(
                self.raw.ch_names, hand_code
            )
            if len(key_ind_dict) == 0:
                print(f'No ACC-keys found in keys: {self.raw.ch_names}')
                continue
            print(f'selected channels: {key_ind_dict}'
                  f'\n\tout of {self.raw.ch_names}')

            # select present acc (aux) variables
            file_data_class = utils_dataManagement.triAxial(
                data=self.raw.samples,
                key_indices=key_ind_dict,
            )

            for acc_side in vars(file_data_class).keys():

                if acc_side in ['left', 'right']:
                    # prevent left-calculations on right-files and viaversa
                    if hand_code != 'bilat':
                        if acc_side != file_side:
                            if self.sub not in self.switched_sides:
                                continue
                            else:  # go on w/ non-matching sides, but invert sides for naming of csv's and plots
                                if acc_side == 'left': save_side = 'R'
                                elif acc_side == 'right': save_side = 'L'
                                f = f + '*'
                        else:
                            if self.sub in self.switched_sides:
                                continue
                            else:  # matching sides, correct left-right acc-sides
                                save_side = acc_side[0].upper()
                    
                    else:  # files recorded unilateral
                        save_side = file_side[0].upper()

                    # PREPROCESS
                    # resample if necessary
                    if self.raw.sample_rate > self.goal_fs:
                        setattr(
                            file_data_class,
                            acc_side,
                            utils_preprocessing.resample(
                                getattr(file_data_class, acc_side),
                                Fs_orig=self.raw.sample_rate,
                                Fs_new=self.goal_fs
                            )
                        )
                    
                    # preprocess data in class
                    procsd_arr, _ = preproc.run_preproc_acc(
                        dat_arr=getattr(file_data_class, acc_side),
                        fs=self.goal_fs,
                        to_detrend=True,
                        to_check_magnOrder=True,
                        to_check_polarity=True,
                        to_remove_outlier=True,
                    )
                    # replace arr in class with processed data
                    setattr(
                        file_data_class,
                        acc_side,
                        procsd_arr
                    )

                    self.data = file_data_class  # store in class to work with in notebook

                    temp_acc, temp_ind = find_blocks.find_active_blocks(
                        acc_arr=getattr(file_data_class, acc_side),
                        fs=self.goal_fs,
                        verbose=True,
                        to_plot=True,
                        plot_orig_fname=f,
                        figsave_dir=os.path.join(
                            proj_dir, 'figures', 'tap_block_plots'
                        ),
                        figsave_name=(
                            f'{self.sub}_{self.state}_'
                            f'{save_side}_blocks_detected'
                        ),
                        to_store_csv=True,
                        csv_dir=os.path.join(
                            proj_dir, 'data', 'tap_block_csvs'
                        ),
                        csv_fname=f'{self.sub_csv_code}{self.sub}_'
                                  f'{self.state}_{save_side}',
                                  # save_side replaced acc_side[0].upper() to correct for swapped acc-sides
                    )




    

if __name__ == '__main__':
    # only executes following code when called via
    # command line, not when loaded in, in another
    # script
    
    uncut_path = utils_dataManagement.find_stored_data_path()

    # check for given subs and states
    if len(sys.argv) == 2:

        with open(sys.argv[1], 'r') as json_data:

            cfg = json.load(json_data)

        if cfg['subs_states'] == 'ALL':
            # get unique sub numbers in UNCUT
            subs = utils_dataManagement.get_unique_subs(uncut_path)

        else:
            # get defined subs
            subs = cfg['subs_states']  # subs given as list instead of dict .keys()
        
        for sub in subs:

            print(f'\nSTART SUB {sub}')

            for state in ['M0S0', 'M0S1', 'M1S0', 'M1S1']:
                print(f'\n\tSTART {state}')
                try:
                    rawAccData(
                        sub=sub,
                        state=state,
                        uncut_path=uncut_path,
                        switched_sides=cfg['side_switch'],
                    )
                except FileNotFoundError:
                    print(f'\t{state} not present for sub{sub}')
    
    elif len(sys.argv) == 3:

        rawAccData(
            sub=sys.argv[1],
            state=sys.argv[2],
            uncut_path=uncut_path,
        )








