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
        model = load_model(f"{get_path_models()}{model}.mdl")
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

# def launch_predict(patient:any, n_experience:int):

#     if patient == "All":
#         score_global = []
#         for subject in range(1,110):
#             print(f"-----> {colors.green}Subject {int(subject)}{colors.reset}", end='')
#             model =  get_name_model(int(subject),n_experience)
#             print(f"\t model: [{colors.blue}{model}{colors.reset}] predict exp= '{colors.yellow}{experiments[n_experience]['description']}{colors.reset}' ", end='')
#             score = predict(subject=subject, n_experience=n_experience, model=model)
#             if score is not None:
#                 score_global.append(score)
#                 print(f" => score = {colorize(score)}")
#             else:
#                 print(f" => Not Trained")
#         print (f"mean Score Global for [{colors.yellow}{experiments[n_experience]['description']}{colors.reset}] with {colors.blue}{len(score_global)}{colors.reset} patient(s) = {colorize(np.mean(score_global))}")
#     else:
#         print(f"Exp= '{colors.yellow}{experiments[n_experience]['description']}{colors.reset}'", end='')
#         # print(f"-----> {colors.green}Subject {int(patient)}{colors.reset}")
#         model =  get_name_model(int(patient),n_experience)
#         print(f"\t model: [{colors.blue}{model}{colors.reset}]", end='')
#         score = predict(subject=patient, n_experience=n_experience, model=model)
#         print(f" => score = {colorize(score)}")