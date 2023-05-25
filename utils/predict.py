import sys
import numpy as np
from sklearn.model_selection import train_test_split, ShuffleSplit, cross_val_score
from sklearn.metrics import accuracy_score

from .experiments import experiments
from .utils_raw import *
from .commun import *

def predict(subject:int, n_experience:int, model=None, verbose = False):
    if verbose:
        print("Process start with parameters : subject=", subject, ", experience=", n_experience, " model=", model)
    
    if model is None:
        model = load_model(get_path(subject, n_experience))
    else:
        model = load_model(f"{get_path_models()}{model}.mdl")
    
    subject = int((subject))
    n_experience = int(n_experience)
    drop_option = model.drop_option
    raw, _ = get_raw(subject = subject, n_experience=n_experience, drop_option=drop_option)
    if drop_option:
        raw = drop_bad_channels(raw=raw, name=get_name_model(subject=subject, n_experience=n_experience),save=False , verbose=False)

    # Apply band-pass filter
    raw = my_filter(raw,verbose=verbose)

    X_train, X_test, y_train, y_test = perso_splitter(raw)

   
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
        print(f'Mean accuracy for all experiments:{colorize(score_subject)}')
        # print(f"Mean cross val score {mean_scores_ldashrinkage}")
    return score_subject

def launch_predict(patient:any, models:list, experiences:list):
    print(f"patient={patient}\nmodels={models}\nexperience={experiences}")
    if patient == "All":
        score_global = []
        for subject in range(1,110):
            print(f"-----> {colors.green}Subject {int(subject)}{colors.reset}")
            score_subject = []
            for m in models:
                score_models = []
                for exp in experiences:
                    model = get_name_model(subject,m)
                    print(f"\t model: [{model}] predict exp={exp} ", end='')
                    if exist(subject=subject, n_experience=exp) or exist(subject=subject,n_experience=m):
                        score = predict(subject=subject, n_experience=exp, model=model, drop_option=drop_option, verbose = False)
                        score_models.append(score)
                        score_subject.append(score)
                        score_global.append(score)
                        print(f" => score = {colorize(score)}")
                    else:
                        print(f" {colors.yellow}Not trained{colors.reset}")
                print (f"mean Score for {model} with {len(score_models)} model(s) and {len(experiences)} exp = {colorize( np.mean(score_models))}")
            print (f"mean Score for subject={subject} with {len(models)} models = {colorize(np.mean(score_subject))}")
            print("----------------------------------------------------")
        print (f"mean Score Global = {colorize(np.mean(score_global))}")
        print(f" Process...{colors.blue} Done.{colors.reset}")
    else:
        print(f"-----> {colors.green}Subject {int(patient)}{colors.reset}")
        score_global = []
        for m in models:
            score_models=[]
            for exp in experiences:
                model = get_name_model(int(patient),m)
                print(f"\t model: [{model}] predict exp={exp} ", end='')
                # if exist(subject=patient, n_experience=exp):
                score = predict(subject=patient, n_experience=exp, model=model, verbose = False)
                score_global.append(score)
                score_models.append(score)
                print(f" => score = {colorize(score)}")
            print (f"mean Score for {model} with {len(score_global)} experiences = {colorize(np.mean(score_models))}")
        print (f"mean Global Score{colorize(np.mean(score_global))}")