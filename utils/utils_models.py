import os
from joblib import dump, load
from .commun import colors, get_json_value


def load_model(path_model):
    if os.path.exists(path_model):
            return load(path_model)  
    return None

def save_model(clf, path_model:str, verbose=False):
    dump(clf, path_model)
    if verbose:
        print(f"Model [{colors.blue}{path_model}{colors.reset}] was saved!")

def get_name_model(subject:int, n_experience:int) -> str:
    name = f"E{n_experience}S{subject:03d}"
    return name

def get_path_model(subject:int, n_experience:int):
    path = f"{get_json_value('MODELS_PATH_DIR')}{get_name_model(subject, n_experience)}.mdl"
    return path