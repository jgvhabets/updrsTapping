
import os
import datetime as dt

import retap_utils.utils_dataManagement as utils_dataMangm
from tap_extract_fts.main_featExtractionClass import FeatureSet  # mandatory for pickle import

if __name__ == '__main__':
    """
    run from main repo path on WINDOWS and macOS as:

        python -m tap_extract_fts.run_main_ftExtraction
    
        -m is there because it is ran from a module/package 
        (within a folder)
    """
    print('...running run_main_ftExtraction.py')
    max_n_taps_incl = 0  # 0 leads to inclusion of all taps 

    data = FeatureSet(
        subs_incl = ['BER026', 'BER056'],
        centers_incl = ['BER', 'DUS'],   # 'DUS'
        verbose=False,
        max_n_taps_incl=max_n_taps_incl,
    )

    deriv_path = os.path.join(
        utils_dataMangm.get_local_proj_dir(),
        'data',
        'derivatives'
    )
    dd = str(dt.date.today().day).zfill(2)
    mm = str(dt.date.today().month).zfill(2)
    yyyy = dt.date.today().year
    
    # to plot for submission figure
    fname = f'ftClass_FIGS_{yyyy}{mm}{dd}'

    # if max_n_taps_incl == 0: fname = f'ftClass_ALL_{yyyy}{mm}{dd}'

    # else: fname = f'ftClass_max{max_n_taps_incl}_{yyyy}{mm}{dd}'
    

    utils_dataMangm.save_class_pickle(
        class_to_save=data,
        path=deriv_path,
        filename=fname,
    )


