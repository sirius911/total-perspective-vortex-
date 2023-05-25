import sys
import numpy as np
from sklearn.model_selection import train_test_split, ShuffleSplit, cross_val_score
from sklearn.metrics import accuracy_score

from .experiments import experiments
from .utils_raw import *
from .commun import *

def predict(subject:int, n_experience:int, model=None, verbose = False):
    subject = int((subject))
    n_experience = int(n_experience)
    
    if model is None:
        model = load_model(get_path(subject, n_experience))
    else:
        model = load_model(f"{MODELS_PATH_DIR}{model}.mdl")
    if model is None:
        return None
    drop_option = model.drop_option
    raw, _ = get_raw(subject = subject, n_experience=n_experience, drop_option=drop_option)
    if drop_option:
        raw = drop_bad_channels(raw=raw, name=get_name_model(subject=subject, n_experience=n_experience),save=False , verbose=False)

    # Apply band-pass filter
    raw = my_filter(raw,verbose=verbose)

    _, X_test, _, y_test = perso_splitter(raw)

    default_stdout = sys.stdout
    # Rediriger la sortie vers null
    sys.stdout = open('/dev/null', 'w')
    predictions = model.predict(X_test)
    #accuracy score
    score_subject = accuracy_score(predictions, y_test)

    # Restaurer la sortie par défaut
    sys.stdout = default_stdout

    return score_subject