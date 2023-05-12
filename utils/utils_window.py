import tkinter as tk
from .utils_raw import get_predict

class Option:
    def __init__(self, parent):
        self.events = tk.BooleanVar()
        self.spectral = tk.BooleanVar()
        self.ica = tk.BooleanVar()
        
        self.events.set(False)
        self.spectral.set(False)
        self.ica.set(False)
        
        self.events_checkbutton = tk.Checkbutton(parent, text="Events", variable=self.events)
        self.spectral_checkbutton = tk.Checkbutton(parent, text="Spectral", variable=self.spectral)
        self.ica_checkbutton = tk.Checkbutton(parent, text="ICA", variable=self.ica)
        
        self.events_checkbutton.pack(side="left")
        self.spectral_checkbutton.pack(side="left")
        self.ica_checkbutton.pack(side="left")

def change_button_analyse(analys_button, patient):
    if patient == 'All':
        analys_button['state'] = tk.DISABLED
    else:
        analys_button['state'] = tk.NORMAL

def change_radiobutton_predict(n_experience, combo, predict_button):
    new_values =  get_predict(n_experience)
    # print(new_values)
    combo['values'] =new_values
    combo.set('Select Patient')
    predict_button['state'] = tk.DISABLED
    
def change_button_predict(predict_button, patient):
    print(f"{patient}")
    if patient is not None or patient != '':
        predict_button['state'] = tk.NORMAL
    else:
        predict_button['state'] = tk.DISABLED