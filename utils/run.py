import tqdm
import numpy as np
from .analyse import analyse
from .train import train
from .predict import predict
from .experiments import experiments
from .commun import colorize, colors
from .utils_models import get_name_model


def launch_process(patient, experience, type_process, drop_option=True, options=None):
    """
        Launch 'type_process' with subject=patient, num of experience = experience and other option
        options is for analyse options display
        drop_option specifies if the wrong channels are dropped
    """
    score = []
    if type_process == 'ANALYSE':
        analyse(patient, experience, drop_option, options=options)
    elif type_process == 'TRAIN':
        if patient == "All":
            for subject in tqdm(range(1, 110)):
                score.append(train(subject, experience, drop_option, csp_name=options, verbose=False))
        else:
            score.append(train(int(patient), experience, drop_option, csp_name=options, verbose=True))
        print (f"mean Score = {colorize(np.mean(score))}")
    elif type_process == 'PREDICT':
        if patient == "All":
            score_global = []
            for subject in range(1,110):
                print(f"-----> {colors.green}Subject {int(subject)}{colors.reset}", end='')
                model =  get_name_model(int(subject), experience)
                print(f"\t model: [{colors.blue}{model}{colors.reset}] predict exp= '{colors.yellow}{experiments[experience]['description']}{colors.reset}' ", end='')
                score = predict(subject=subject, n_experience=experience, model=model)
                if score is not None:
                    score_global.append(score)
                    print(f" => score = {colorize(score)}")
                else:
                    print(f" => Not Trained")
            print (f"mean Score Global for [{colors.yellow}{experiments[experience]['description']}{colors.reset}] with {colors.blue}{len(score_global)}{colors.reset} patient(s) = {colorize(np.mean(score_global))}")
        else:
            print(f"Exp= '{colors.yellow}{experiments[experience]['description']}{colors.reset}'", end='')
            model =  get_name_model(int(patient), experience)
            print(f"\t model: [{colors.blue}{model}{colors.reset}]", end='')
            score = predict(subject=patient, n_experience=experience, model=model)
            print(f" => score = {colorize(score)}")
    print(f"\n\t\tProcess ...{colors.blue}Done{colors.reset}")
    print("---------------------------------------------------")  
    return