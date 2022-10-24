"""
Utilisation functions as part of (updrsTapping-repo)
ReTap-Toolbox
"""

# import public packages and functions
import os


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


def get_arr_key_indices(ch_names):

    dict_out = {}

    for i, key in enumerate(ch_names):

        if key in ['X', 'Y', 'Z']:

            if f'L_{key}' in dict_out.keys():

                dict_out[f'R_{key}'] = i
            
            else:

                dict_out[f'L_{key}'] = i

    return dict_out