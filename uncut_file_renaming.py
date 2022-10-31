"""
Renaming function

Run in python to standardise the medstim-state
coding of the acc files.
Perform before cutting so the cutted blocks
have the correct M0S0 naming
"""

# import necessary functions
import os

from retap_utils import utils_dataManagement

uncut_path = utils_dataManagement.find_stored_data_path()

state_strings = {
    'M0S0': [
        'MedOffStimOff',
        'MOffSOff',
        'medOff_stimOff'
    ],
    'M0S1': [
        'MedOffStimOn',
        'MOffSOn',
        'medOff_stimOn'
    ],
    'M1S0': [
        'MedOnStimOff',
        'MOnSOff',
        'medOn_stimOff'
    ],
    'M1S1': [
        'MedOnStimOn',
        'MOnSOn',
        'medOn_stimOn'
    ]
}


for f in os.listdir(uncut_path):

    for state in state_strings:

        for string in state_strings[state]:
            if string in f:
                f_new = f.replace(string, state)
                print('rename', f)
                os.rename(
                    os.path.join(uncut_path, f),
                    os.path.join(uncut_path, f_new)
                )
                print('\tinto', f_new)

