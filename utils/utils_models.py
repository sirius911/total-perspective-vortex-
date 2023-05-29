import os
from joblib import dump, load
from .commun import colors


def load_model(path_model):
    if os.path.exists(path_model):
            return load(path_model)  
    return None

def save_model(clf, path_model:str, verbose=False):
    dump(clf, path_model)
    if verbose:
        print(f"Model [{colors.blue}{path_model}{colors.reset}] was saved!")
