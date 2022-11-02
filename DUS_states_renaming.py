"""
Renaming function

Run in python to standardise the medstim-state
coding of the acc files.
Perform before cutting so the cutted blocks
have the correct M0S0 naming
"""
if __name__ == '__main__':
    # import necessary functions
    import os

    from retap_utils import utils_dataManagement

    from pandas import read_csv

    path = utils_dataManagement.find_stored_data_path('DUS')
    
    for f in os.listdir(path):

        if f[-3:] != 'csv': continue

        sub = f.split('_')[0]
        med = f.split('_')[1]
        stim = f.split('_')[3]

        if med == 'OFF': med = 'M0'
        elif med == 'ON': med = 'M1'

        if 'nostim' in stim.lower(): stim = 'S0'
        else: stim = 'S1'

        f_new = f'{sub}_{med}{stim}.csv'

        data = read_csv(os.path.join(path, f))

        data.to_csv(os.path.join(path, 'RENAMED', f_new))

        print(f'saved {f_new}')
