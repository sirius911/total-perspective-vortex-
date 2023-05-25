import os
import tkinter as tk
from utils.commun import SAVE_PATH, MODELS_PATH_DIR, BAD_CHANNELS_DIR, colors
from utils.utils_window import create_window
from utils.menu import welcome


def main_window():
    # Create Main window
    window = tk.Tk()
    window = create_window(window)
    
    # Launching the event loop of the window
    window.mainloop()

if __name__ == "__main__":
    print(welcome)
    print(f"Backup  Folder : {colors.blue}{SAVE_PATH}{colors.reset}")
    print(f"\tModels : {colors.blue}{MODELS_PATH_DIR}{colors.reset}")
    print(f"\tBad channels : {colors.blue}{BAD_CHANNELS_DIR}{colors.reset}")
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
    if not os.path.exists(MODELS_PATH_DIR):
        os.makedirs(MODELS_PATH_DIR)
    if not os.path.exists(BAD_CHANNELS_DIR):
        os.makedirs(BAD_CHANNELS_DIR)
    main_window()
    print("Good bye !")

