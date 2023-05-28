import os

SAVE_PATH = (os.path.dirname(os.path.abspath('main.py'))+"/save")
PATH_DATA = os.getenv('HOME') + '/sgoinfre'
MODELS_PATH_DIR = SAVE_PATH+"/models/"
BAD_CHANNELS_DIR = SAVE_PATH+"/bad_channels/"
ALGO = "algo"
ALGO_PATH = (os.path.dirname(os.path.abspath('main.py'))+"/"+ALGO)

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