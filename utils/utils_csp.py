import os
import json
import importlib
from .commun import colors, get_json_value

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
    if csp_name != "mne.decoding.CSP":
        module_name = csp_name
        csp_module = importlib.import_module(module_name)
        CSP = csp_module.CSP
    else:
        from mne.decoding import CSP
    n_components, log = load_param_csp(csp_name)
    return(CSP(n_components=n_components, log=log))
