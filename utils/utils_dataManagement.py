"""
Utilisation functions as part of (updrsTapping-repo)
ReTap-Toolbox
"""

# import public packages and functions
import os
import numpy as np


def get_proj_dir():
    """
    Takes parent-directory until main-project
    folder is found, containing code/, data/,
    figures/
    """
    dir = os.getcwd()

    while dir[-4:] != 'code':

        dir = os.path.dirname(dir)

    proj_dir = os.path.dirname(dir)

    return proj_dir


def get_file_selection(
    path, sub, state,
    joker_string = None
):
    sel_files = []
        
    for f in os.listdir(path):

        if np.logical_and(
            f'sub{sub}' not in f.lower(),
            f'sub{sub[1:]}' not in f.lower(),
        ): continue

        if type(joker_string) == str:

            if joker_string not in f:

                continue

        if state in f:

            sel_files.append(f)
    
    return sel_files


def get_arr_key_indices(ch_names):

    dict_out = {}

    aux_keys = [
        'L_X', 'L_Y', 'L_Z',
        'R_X', 'R_Y', 'R_Z'
    ]
    aux_count = 0

    for i, key in enumerate(ch_names):

        if key in ['X', 'Y', 'Z']:

            if f'L_{key}' in dict_out.keys():

                dict_out[f'R_{key}'] = i
            
            else:

                dict_out[f'L_{key}'] = i
        
        elif 'aux' in key.lower():

            dict_out[aux_keys[aux_count]] = i
            aux_count += 1


    return dict_out