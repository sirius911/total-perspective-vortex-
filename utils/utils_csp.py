import os
import json
import importlib
from .commun import colors, get_json_value, ALGO

def load_param_csp(csp_name:str):
    """
        return param of csp from algo/csp_param.json
    """
    n_components = 6
    log = True
    path_config = get_json_value("ALGO_PATH")+"/csp_param.json"
    try:
        if os.path.exists(path_config):
            with open(path_config, 'r') as file:
                data = json.load(file)
                n_components = data[csp_name]['n_components']
                log = data[csp_name]['log']
    except Exception:
        print(f"{colors.red}Warning:{colors.reset} No param in {colors.yellow}{path_config}{colors.reset} --> Param defaut (n={n_components}, log={log})")
        pass
    return n_components, log   

def get_csp(csp_name:str):
    """
        Return the CSP with csp_name in algo/ with the param in algo/csp_param.json
    """
    if csp_name != "mne.decoding.CSP":
        module_name = csp_name
        csp_module = importlib.import_module(module_name)
        CSP = csp_module.CSP
    else:
        from mne.decoding import CSP
    n_components, log = load_param_csp(csp_name)
    return(CSP(n_components=n_components, log=log))

def load_algo(algo_path = get_json_value("ALGO_PATH")) -> list:
    """
        return the list of csp algo in the folder algo_path
    """
    files_py = []
    files = os.listdir(algo_path)
    for file in files:
        path = os.path.join(algo_path, file)
        if file.endswith(".py") and os.path.isfile(path) and not (file.startswith('__')):
            csp_name = ALGO +"." + os.path.basename(os.path.splitext(file)[0])
            files_py.append(csp_name)
    return files_py
