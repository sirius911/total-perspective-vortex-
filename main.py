import os
import tkinter as tk
from utils.commun import colors, get_json_value, create_path_json
from utils.utils_window import create_window
from utils.menu import welcome

def prepare_folders():
    if not os.path.exists("utils/path.json"):
        create_path_json()
    
    if not os.path.exists(get_json_value('SAVE_PATH')):
        os.makedirs(get_json_value('SAVE_PATH'))
    if not os.path.exists(get_json_value('MODELS_PATH_DIR')):
        os.makedirs(get_json_value('MODELS_PATH_DIR'))
    if not os.path.exists(get_json_value('BAD_CHANNELS_DIR')):
        os.makedirs(get_json_value("BAD_CHANNELS_DIR"))
    
    print(f"Backup  Folder : {colors.blue}{get_json_value('SAVE_PATH')}{colors.reset}")
    print(f"\tModels : {colors.blue}{get_json_value('MODELS_PATH_DIR')}{colors.reset}")
    print(f"\tBad channels : {colors.blue}{get_json_value('BAD_CHANNELS_DIR')}{colors.reset}")
    print(f"\tPhysionet's Data : {colors.blue}{get_json_value('PATH_DATA')}{colors.reset}")
    print(f"\tCSP algorythms folder : {colors.blue}{get_json_value('ALGO_PATH')}{colors.reset}")   

def main_window():
    # Create Main window
    window = tk.Tk()
    window = create_window(window)
    
    # Launching the event loop of the window
    window.mainloop()

if __name__ == "__main__":
    print(welcome)
    prepare_folders()
    main_window()
    print("Good bye !")

