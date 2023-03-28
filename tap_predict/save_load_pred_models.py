"""
Save and Load Cross-Validation prediction models
for HOLD OUT VALIDATION.
"""

# import public packages
from os.path import join, exists
from os import makedirs
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from joblib import dump, load

from retap_utils.utils_dataManagement import get_local_proj_dir
from tap_plotting.plot_pred_results import plot_ft_importances

def save_model_in_cv(
    clf, X_CV, y_CV, model_fname,
    path=None, random_state=27,
    to_plot_ft_importances=False,
    ft_names=None, ADD_FIG_PATH=None,
):
    if not path: path = join(get_local_proj_dir(), 'results', 'models')
    # create path if not existing
    if not exists(path): makedirs(path)
    # check pickle extension
    if not model_fname.endswith('P') or model_fname.endswith('pkl'):
        model_fname += '.P'

    # set random state    
    np.random.seed(random_state)

    # retrain model on ALL cross-validation data
    if clf.upper() == 'RF':
        CLF = RandomForestClassifier(
            n_estimators=1000,  # 500
            max_depth=None,
            min_samples_split=5,
            max_features='sqrt',
            random_state=random_state,
            class_weight='balanced',
        )
    
    elif clf.upper() == 'LOGREG':
        CLF = LogisticRegression(random_state=random_state, solver='lbfgs',)

    elif clf.upper() == 'SVC':
        CLF = SVC(kernel='linear', class_weight='balanced',
                  gamma='scale', random_state=27,)

    # fit model
    CLF.fit(X=X_CV, y=y_CV)

    if to_plot_ft_importances:
        if clf.upper() == 'RF':
            plot_ft_importances(fitted_clf=CLF,
                                ft_names=ft_names,
                                model_name=model_fname[:-2],
                                ADD_FIG_PATH=ADD_FIG_PATH)

    # save the model as pickle
    dump(CLF, join(path, model_fname))

    check_load = load(join(path, model_fname))

    print(f'model succesfully saved as {join(path, model_fname)}'
            '; loading succesfully tested')

