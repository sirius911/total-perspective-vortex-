import os

SAVE_PATH = (os.path.dirname(os.path.abspath('main.py'))+"/save")
PATH_DATA = os.getenv('HOME') + '/goinfre'

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