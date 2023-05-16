import sys
import numpy as np
from sklearn.model_selection import train_test_split, ShuffleSplit, cross_val_score
from sklearn.metrics import accuracy_score

from .experiments import experiments
from .utils_raw import *

def predict(subject:int, n_experience:int, model=None, drop_option= True, verbose = False):
    if verbose:
        print("Process start with parameters : subject=", subject, ", experience=", n_experience, " model=", model)
    
    subject = int((subject))
    n_experience = int(n_experience)
    runs = experiments[n_experience]['runs']
    raw, _ = get_raw(subject = subject, n_experience=n_experience, runs=runs, drop_option=drop_option)
    if drop_option:
        raw = raw = drop_bad_channels(raw=raw, name=get_name_model(subject=subject, n_experience=n_experience),save=False , verbose=verbose)

    # Apply band-pass filter
    raw = my_filter(raw,verbose=verbose)

    cv = ShuffleSplit(10, test_size=0.2, random_state=0)

    # Read epochs
    epochs_train, labels = get_data(raw)
    _, X_test, _, y_test = train_test_split(epochs_train, labels, random_state=0)

    if model is None:
        model = load_model(get_path(subject, n_experience))
    else:
        model = load_model(f"{get_path_models()}{model}.mdl")
    if verbose == False:
        default_stdout = sys.stdout
        # Rediriger la sortie vers null
        sys.stdout = open('/dev/null', 'w')
    predictions = model.predict(X_test)
    #accuracy score
    score_subject = accuracy_score(predictions, y_test)
    #cross_val_score
    # score_subject = np.mean(cross_val_score(model, epochs_train, labels, cv=cv, n_jobs=-1, verbose=False))


    if verbose == False:
        # Restaurer la sortie par défaut
        sys.stdout = default_stdout
    else:
        print(f'event nb: [prediction] [truth] equal?')
        for i, prediction in enumerate(predictions):
            print(f'event {i:02d}: [{prediction}] [{y_test[i]}] {valid(prediction == y_test[i])}')
        print(f'Mean accuracy for all experiments:{score_subject}')
        # print(f"Mean cross val score {mean_scores_ldashrinkage}")
    return score_subject

def launch_predict(patient:any, models:list, experiences:list, drop_option=True):
    if patient == "All":
        score_global = []
        for subject in range(1,110):
            print(f"-----> {colors.green}Subject {int(subject)}{colors.reset}")
            score_subject = []
            for m in models:
                score_models = []
                for exp in experiences:
                    model = get_name_model(subject,m)
                    print(f"\t model: [{model}]", end='')
                    if exist(subject=subject, n_experience=exp) or exist(subject=subject,n_experience=m):
                        score = predict(subject=subject, n_experience=exp, model=model, drop_option=drop_option, verbose = False)
                        score_models.append(score)
                        score_subject.append(score)
                        score_global.append(score)
                        print(f" => score = {score}")
                    else:
                        print(f" {colors.yellow}Not trained{colors.reset}")
                print (f"mean Score for {model} with {len(score_models)} model(s) and {len(experiences)} exp = {np.mean(score_models)}")
            print (f"mean Score for subject={subject} with {len(models)} models = {np.mean(score_subject)}")
            print("----------------------------------------------------")
        print (f"mean Score Global = {np.mean(score_global)}")
        print(f" Process...{colors.blue} Done.{colors.reset}")
    else:
        print(f"-----> {colors.green}Subject {int(patient)}{colors.reset}")
        score_global = []
        for m in models:
            for exp in experiences:
                model = get_name_model(int(patient),m)
                print(f"\t model: [{model}]", end='')
                # if exist(subject=patient, n_experience=exp):
                score = predict(subject=patient, n_experience=exp, model=model, drop_option=drop_option, verbose = False)
                score_global.append(score)
                print(f" => score = {score}")
            print (f"mean Score for {model} with {len(score_global)} experiences = {np.mean(score_global)}")