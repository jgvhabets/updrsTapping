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

def save_model_in_cv(
    clf, X_CV, y_CV, model_fname,
    path=None, random_state=27,
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
    if clf == 'RF':
        clf = RandomForestClassifier(
            n_estimators=1000,  # 500
            max_depth=None,
            min_samples_split=5,
            max_features='sqrt',
            random_state=random_state,
            class_weight='balanced',
        )
    
    elif clf == 'LogReg':
        clf = LogisticRegression(random_state=random_state, solver='lbfgs',)

    # fit model
    clf.fit(X=X_CV, y=y_CV)

    # save the model as pickle
    dump(clf, join(path, model_fname))

    check_load = load(join(path, model_fname))

    print(f'model succesfully saved as {join(path, model_fname)}'
            '; loading succesfully tested')

