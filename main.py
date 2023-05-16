import os
import tkinter as tk
from utils.commun import SAVE_PATH
from utils.utils_window import create_window


def main_window():
    # Create Main window
    window = tk.Tk()
    window = create_window(window)
    
    # Launching the event loop of the window
    window.mainloop()

if __name__ == "__main__":
    print(SAVE_PATH)
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
    models_path = (SAVE_PATH+"/models/")
    if not os.path.exists(models_path):
        os.makedirs(models_path)
    bad_channels=SAVE_PATH+"/bad_channels/"
    if not os.path.exists(bad_channels):
        os.makedirs(bad_channels)
    main_window()
    print("Good bye !")

