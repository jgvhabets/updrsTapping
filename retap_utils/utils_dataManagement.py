"""
Utilisation functions as part of (updrsTapping-repo)
ReTap-Toolbox
"""

# import public packages and functions
import os
from sys import platform
import numpy as np
from dataclasses import dataclass
from array import array


def get_proj_dir():
    """
    Device and OS independent function to find
    the main-project folder, where this repo is
    stored. Project folder should contain subfolders:
    code (containinng this repo), data, figures
    """
    dir = os.getcwd()

    while dir[-4:] != 'code':

        dir = os.path.dirname(dir)

    proj_dir = os.path.dirname(dir)

    return proj_dir


def find_stored_data_path(
    folder: str
):
    """
    Device and OS independent function to find
    the synced-OneDrive folder where data is stored
    """
    if folder.lower() not in [
        'onedrive', 'retapdata', 'dus', 'ber','uncut'
    ]:
        raise ValueError(
            f'given folder: {folder} is incorrect, '
            'should be [onedrive, retapdata, BER, or UNCUT')
        
    path = os.getcwd()
    while os.path.dirname(path)[-5:] != 'Users':
        path = os.path.dirname(path)
    # path is now Users/username
    onedrive_f = [
        f for f in os.listdir(path) if np.logical_and(
            'onedrive' in f.lower(),
            'charit' in f.lower()
        ) 
    ]
    onedrive = os.path.join(path, onedrive_f[0])
    if folder.lower() == 'onedrive': return onedrive

    retapdata = os.path.join(onedrive, 'ReTap', 'data')
    if folder.lower() == 'retapdata': return retapdata

    DUSdata = os.path.join(retapdata, 'dus', 'RENAMED')
    if folder.lower() == 'dus': return DUSdata
    
    BERdata = os.path.join(retapdata, 'BER')
    if folder.lower() == 'ber': return BERdata

    uncut = os.path.join(BERdata, 'UNCUT')
    if folder.lower() == 'uncut': return uncut
    

def get_unique_subs(path):

    files = os.listdir(path)
    subs = [
        f.split('_')[0][-3:] for f in files
        if f[:3].lower() == 'sub'
    ]
    subs = list(set(subs))

    return subs


def get_file_selection(
    path, sub, state,
    joker_string = None
):
    sel_files = []
        
    for f in os.listdir(path):

        if not np.array([
            f'sub{sub}' in f.lower() or
            f'sub{sub[1:]}' in f.lower() or
            f'sub-{sub}' in f.lower()
        ]).any(): continue

        if type(joker_string) == str:

            if joker_string not in f:

                continue

        if state.lower() in f.lower():

            sel_files.append(f)
    
    return sel_files


def get_arr_key_indices(ch_names, hand_code):
    """
    creates dict with acc-keynames and indices

    assumes that acc-channels are called X, Y, Z
    and that the first three are for the left-finger,
    last three for the right-finger

    Exception possible for only three acc-sensors present

    TODO: CONSIDER GENERAL USABILITY WITH CONFIG-FILE OF ACC-NAMES
    """
    dict_out = {}

    if hand_code == 'bilat':
        
        side = 'bilat'
        aux_keys = [
            'L_X', 'L_Y', 'L_Z',
            'R_X', 'R_Y', 'R_Z'
        ]
    
    else:

        if 'L' in hand_code:
            S = 'L'
            side = 'left'
        elif 'R' in hand_code:
            S = 'R'
            side = 'right'

        aux_keys = [
            f'{S}_X', f'{S}_Y', f'{S}_Z'
        ]

    
    aux_count = 0

    for i, key in enumerate(ch_names):
        # only use keys X-Y-Z
        if key in ['X', 'Y', 'Z']:

            if f'L_{key}' in dict_out.keys():
                # if left exists, make right
                dict_out[f'R_{key}'] = i
            
            else:
                # if left doesnt exist, make left
                dict_out[f'L_{key}'] = i
        
        elif 'aux' in key.lower():

            if 'iso' in key.lower(): continue

            dict_out[aux_keys[aux_count]] = i
            aux_count += 1

    return dict_out, side


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

