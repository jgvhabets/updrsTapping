"""
Utilisation functions as part of (updrsTapping-repo)
ReTap-Toolbox
"""

# import public packages and functions
import os
from pandas import read_excel
import numpy as np
from dataclasses import dataclass
from array import array
import pickle


def get_local_proj_dir():
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


def find_onedrive_path(
    folder: str
):
    """
    Device and OS independent function to find
    the synced-OneDrive folder where data is stored
    """
    folder_options = [
        'onedrive', 'retapdata', 'dus', 'ber','uncut',
        'figures',
    ]
    if folder.lower() not in folder_options:
        raise ValueError(
            f'given folder: {folder} is incorrect, '
            f'should be {folder_options}')
        
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

    figpath = os.path.join(onedrive, 'Retap', 'figures')
    if folder.lower() == 'figures': return figpath
    

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

    # acc_keys = [
    #     'R_X', 'R_Y', 'R_Z',
    #     'L_X', 'L_Y', 'L_Z'
    # ]
    # set laterality of file for later analysis flow
    if hand_code == 'bilat': file_side = 'bilat'
    elif 'L' in hand_code: file_side = 'left'
    elif 'R' in hand_code: file_side = 'right'

    # name acc which are called XYZ or aux without laterality
    aux_count = 0

    for i, key in enumerate(ch_names):
        # standard BER acc-coding is X-Y-Z (first right, then left)
        if key in ['X', 'Y', 'Z']:

            if f'R_{key}' in dict_out.keys():
                # if right exists, make LEFT
                dict_out[f'L_{key}'] = i
            
            else:
                # start with RIGHT keys (first in TMSi files)
                dict_out[f'R_{key}'] = i
        
        elif 'aux' in key.lower():

            if 'iso' in key.lower(): continue  # ISO is no ACC-channel in TMSi

            if 'L' in hand_code: aux_keys = ['L_X', 'L_Y', 'L_Z']
            elif 'R' in hand_code: aux_keys = ['R_X', 'R_Y', 'R_Z']

            dict_out[aux_keys[aux_count]] = i
            aux_count += 1

    return dict_out, file_side


@dataclass(init=True, repr=True)
class triAxial:
    """
    Select accelerometer keys

    TODO for laterb TOOLBOX: add Cfg variable to indicate
    user-specific accelerometer-key
    """
    data: array
    key_indices: dict

    def __post_init__(self,):

        try:
            self.left = self.data[
                self.key_indices['L_X']:
                self.key_indices['L_Z'] + 1  # +1 to include last index while slicing
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
    p = find_onedrive_path('retapdata')
    xl_fname = 'ReTap_participantLog.xlsx'

    log = read_excel(
        os.path.join(p, xl_fname),
        sheet_name=center
    )

    return log


def save_class_pickle(
    class_to_save,
    path,
    filename,
    extension='.P',
):
    pickle_path = os.path.join(
        path, filename + extension
    )

    with open(pickle_path, 'wb') as f:
        pickle.dump(class_to_save, f)
        f.close()

    return print(f'inserted class saved as {pickle_path}')


def load_class_pickle(
    file_to_load,
):
    """
    Loads saved Classes. When running this code
    the class-definitions have to be called before
    executign this code.

    So, for example:

    from tap_extract_fts.main_featExtractionClass import FeatureSet, singleTrace
    from retap_utils import utils_dataManagement as utilsDataMan

    deriv_path = os.path.join(utilsDataMan.get_local_proj_dir(), 'data', 'derivatives')
    loaded_class = utilsDataMan.load_class_pickle(os.path.join(deriv_path, 'classFileName.P'))


    Input:
        - file_to_load: string including path,
            filename, and extension
    
    Returns:
        - output: variable containing the class
    """

    with open(file_to_load, 'rb') as f:
        output = pickle.load(f)
        f.close()

    return output