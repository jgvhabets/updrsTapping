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
    # DUS path is already set to RENAMED FOLDER
    print(f'found DUS parth is {path}')

    for f in os.listdir(path):
        
        if f[-3:] != 'csv': continue
        """RENAME MED-STIM CODING"""
        # sub = f.split('_')[0]
        # med = f.split('_')[1]
        # stim = f.split('_')[3]

        # if med == 'OFF': med = 'M0'
        # elif med == 'ON': med = 'M1'

        # if 'nostim' in stim.lower(): stim = 'S0'
        # else: stim = 'S1'

        # f_new = f'{sub}_{med}{stim}.csv'

        # data = read_csv(os.path.join(path, f))

        # data.to_csv(os.path.join(path, 'RENAMED', f_new))

        # print(f'saved {f_new}')

        """ADD LEFT-SIDE; BLOCK-1 AND HZ"""
        if 'Hz' in f: continue
        if 'block' in f: continue
        if '_L_' in f or '_R_' in f: continue

        f_new = f.split('.')[0]
        f_new += '_L_block1_250Hz.csv'

        os.rename(
            os.path.join(path, f),
            os.path.join(path, f_new)
        )
        print(f'renamed {f_new} (was {f})')
