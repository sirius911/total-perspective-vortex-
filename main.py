import os
import tkinter as tk
from utils.commun import *
from utils.utils_window import create_window
from utils.menu import welcome

def prepare_folders():
    if not os.path.exists("utils/path.json"):
        create_path_json()
    
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
    if not os.path.exists(MODELS_PATH_DIR):
        os.makedirs(MODELS_PATH_DIR)
    if not os.path.exists(BAD_CHANNELS_DIR):
        os.makedirs(BAD_CHANNELS_DIR)
    
    print(f"Backup  Folder : {colors.blue}{SAVE_PATH}{colors.reset}")
    print(f"\tModels : {colors.blue}{MODELS_PATH_DIR}{colors.reset}")
    print(f"\tBad channels : {colors.blue}{BAD_CHANNELS_DIR}{colors.reset}")
    print(f"\tPhysionet's Data : {colors.blue}{PATH_DATA}{colors.reset}")
    print(f"\tCSP algorythms folder : {colors.blue}{ALGO_PATH}{colors.reset}")   

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

