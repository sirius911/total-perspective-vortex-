import os
import json

ALGO = "algo"

def get_default_path():
    save_path = (os.path.dirname(os.path.abspath('main.py'))+"/save")
    data = {
    "SAVE_PATH": save_path,
    "PATH_DATA": os.getenv('HOME') + '/sgoinfre',
    "MODELS_PATH_DIR": save_path+"/models/",
    "BAD_CHANNELS_DIR": save_path+"/bad_channels/",
    "ALGO_PATH": (os.path.dirname(os.path.abspath('main.py'))+"/"+ALGO),
    }
    return data

def get_data_path(file_path="utils/path.json"):
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            data = json.load(file)
    else:
        data = get_default_path()
    return data

def get_json_value(key, file_path = "utils/path.json") -> str:
    if os.path.exists("utils/path.json"):
        with open(file_path, 'r') as file:
            data = json.load(file)
    else:
        data = get_default_path()
    return data.get(key)

SAVE_PATH = get_json_value("SAVE_PATH")
PATH_DATA = get_json_value("PATH_DATA")
MODELS_PATH_DIR = get_json_value("MODELS_PATH_DIR")
BAD_CHANNELS_DIR = get_json_value("BAD_CHANNELS_DIR")
ALGO_PATH = get_json_value("ALGO_PATH")

class colors:
    green = '\033[92m' # vert
    blue = '\033[94m' # blue
    yellow = '\033[93m' # jaune
    red = '\033[91m' # rouge
    reset = '\033[0m' #gris, couleur normales

def valid(test):
    result=""
    if test:
        result = (f"{colors.green}Ok")
    else:
        result = (f"{colors.red}Ko")
    result += (f"{colors.reset}")
    return result

def colorize(percentage: float) -> str:
    if percentage is None:
        return 'None'
    if percentage >= 0.60:
        return colors.green + f'{percentage:.2%}' + colors.reset
    if percentage >= 0.40:
        return colors.yellow + f'{percentage:.2%}' + colors.reset
    return colors.red + f'{percentage:.2%}' + colors.reset

def create_path_json(file_path = "utils/path.json"):
    data = get_default_path()
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)