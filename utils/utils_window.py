import tkinter as tk
from tkinter import ttk
import numpy as np
from tqdm import tqdm

from .utils_raw import get_predict
from .analyse import analyse
from .train import train
from .predict import predict

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
    if len(new_values) == 0:
        combo.set('No models')
        combo['state'] = tk.DISABLED
    else:
        combo.set('Select Patient')
        combo['state'] = tk.NORMAL
    predict_button['state'] = tk.DISABLED
    

def change_button_predict(predict_button, patient):
    if patient is not None or patient != '':
        predict_button['state'] = tk.NORMAL
    else:
        predict_button['state'] = tk.DISABLED

def launch_process(patient, experience, type_process, drop_option=True, options=None, liste=[]):
    score = []
    if type_process == 'ANALYSE':
        analyse(patient, experience, drop_option, options=options)
    elif type_process == 'TRAIN':
        if patient == "All":
            for subject in tqdm(range(1, 110)):
                score.append(train(subject, experience, drop_option, verbose=False))
        else:
            score.append(train(int(patient), experience, drop_option, verbose=True))
        print (f"mean Score = {np.mean(score)}")
    elif type_process == "PREDICT":

        if patient == "All":
            for subject in tqdm(liste):
                score.append(predict(subject, experience, drop_option, verbose=False))
        else:
            score.append(predict(int(patient), experience, drop_option, verbose=True))
        print (f"mean Score = {np.mean(score)}")
    return

def create_window() -> tk:
     # Create Main window
    window = tk.Tk()
    window.title("PhysioNet / EEG")

    onglets = ttk.Notebook(window)

    # Création des onglets
    onglet_analyse = ttk.Frame(onglets)
    onglet_training = ttk.Frame(onglets)
    onglet_predict = ttk.Frame(onglets)

    # Ajout des onglets au widget Notebook
    onglets.add(onglet_analyse, text="Analyse")
    onglets.add(onglet_training, text="Training")
    onglets.add(onglet_predict, text="Predict")

    ###### ONGLET ANALYSE #######
    # Framework for patient choices
    patient_frame_analyse = tk.LabelFrame(onglet_analyse, text="Patient")
    patient_frame_analyse.pack(padx=10, pady=10)

    patient_analyse_var = tk.StringVar()
    patient_analyse_var.set("Set Patient")
    patients = list(range(1, 110))
    # patients = []
    # patients.append("All")
    # patients.extend(range(1, 110))

    patient_analyse_combo = ttk.Combobox(patient_frame_analyse, textvariable=patient_analyse_var, values=patients, state="readonly")
    patient_analyse_combo.pack(padx=10, pady=10)
    # Framework for experience choices
    experience_frame = tk.LabelFrame(onglet_analyse, text="Experience")
    experience_frame.pack(padx=10, pady=10)
    experience_var = tk.IntVar(value=0)

    tk.Radiobutton(experience_frame, text="Open and close left or right Fist", variable=experience_var, value=0).pack(anchor="w")
    tk.Radiobutton(experience_frame, text="Imagine opening and closing left or right Fist", variable=experience_var, value=1).pack(anchor="w")
    tk.Radiobutton(experience_frame, text="Open and close both Fists or both Feets", variable=experience_var, value=2).pack(anchor="w")
    tk.Radiobutton(experience_frame, text="Imagine opening and closing both Fists or both Feets", variable=experience_var, value=3).pack(anchor="w")
    tk.Radiobutton(experience_frame, text="Movement (Real or Imagine) of fists", variable=experience_var, value=4).pack(anchor="w")
    tk.Radiobutton(experience_frame, text="Movement (Real or Imagine) of Fists or Feets", variable=experience_var, value=5).pack(anchor="w")

    # Checkbox for Drop bad channels option
    drop_option =  tk.BooleanVar(value=True)
    drop_checkbutton = tk.Checkbutton(onglet_analyse, text="Drop Bad Channels", variable=drop_option)
    drop_checkbutton.pack(padx=10, pady=10)

    #frames
    options_analyse_frame = tk.LabelFrame(onglet_analyse, text="Options")
    options_analyse_frame.pack(padx=10, pady=10)

    options = Option(options_analyse_frame)

    # button Analyse
    analys_button = tk.Button(onglet_analyse, text="Launch the analysis", state="disabled", command=lambda:launch_process(patient_analyse_var.get(), experience_var.get(), type_process='ANALYSE', drop_option=drop_option.get(), options=options))
    analys_button.pack(padx=10, pady=10)

    ####### ONGLET TRAIN ########

    # Framework for patient choices
    patient_frame_train = tk.LabelFrame(onglet_training, text="Patient")
    patient_frame_train.pack(padx=10, pady=10)

    patient_train_var = tk.StringVar()
    patient_train_var.set("All")
    patients = []
    patients.append("All")
    patients.extend(range(1, 110))

    patient_train_combo = ttk.Combobox(patient_frame_train, textvariable=patient_train_var, values=patients, state="readonly")
    patient_train_combo.pack(padx=10, pady=10)
    
    # Framework for experience choices
    experience_train = tk.LabelFrame(onglet_training, text="Experience")
    experience_train.pack(padx=10, pady=10)

    experience_train_var = tk.IntVar(value=0)
    tk.Radiobutton(experience_train, text="Open and close left or right Fist", variable=experience_train_var, value=0).pack(anchor="w")
    tk.Radiobutton(experience_train, text="Imagine opening and closing left or right Fist", variable=experience_train_var, value=1).pack(anchor="w")
    tk.Radiobutton(experience_train, text="Open and close both Fists or both Feets", variable=experience_train_var, value=2).pack(anchor="w")
    tk.Radiobutton(experience_train, text="Imagine opening and closing both Fists or both Feets", variable=experience_train_var, value=3).pack(anchor="w")
    tk.Radiobutton(experience_train, text="Movement (Real or Imagine) of fists", variable=experience_train_var, value=4).pack(anchor="w")
    tk.Radiobutton(experience_train, text="Movement (Real or Imagine) of Fists or Feets", variable=experience_train_var, value=5).pack(anchor="w")
   
    #button train
    train_button = tk.Button(onglet_training, text="Train", command=lambda:launch_process(patient_train_var.get(), experience_train_var.get(), type_process='TRAIN', drop_option=drop_option.get()))
    train_button.pack(padx=10, pady=10)

    ###### ONGLET PREDICT #######

    predict_frame = tk.LabelFrame(onglet_predict, text="Experiences")
    predict_frame.pack(padx=2, pady=2)
    patient_predict_combo=None
    experience_predict_var = tk.IntVar(value=0)
    rb1=tk.Radiobutton(predict_frame, text="Open and close left or right Fist", variable=experience_predict_var, value=0, command = lambda : change_radiobutton_predict(0, patient_predict_combo, predict_button))
    rb1.pack(anchor="w")
    rb2=tk.Radiobutton(predict_frame, text="Imagine opening and closing left or right Fist", variable=experience_predict_var, value=1, command = lambda : change_radiobutton_predict(1, patient_predict_combo, predict_button))
    rb2.pack(anchor="w")
    rb3=tk.Radiobutton(predict_frame, text="Open and close both Fists or both Feets", variable=experience_predict_var, value=2,command = lambda : change_radiobutton_predict(2, patient_predict_combo, predict_button))
    rb3.pack(anchor="w")
    rb4=tk.Radiobutton(predict_frame, text="Imagine opening and closing both Fists or both Feets", variable=experience_predict_var, value=3, command = lambda : change_radiobutton_predict(3, patient_predict_combo, predict_button))
    rb4.pack(anchor="w")
    rb5=tk.Radiobutton(predict_frame, text="Movement (Real or Imagine) of fists", variable=experience_predict_var, value=4, command = lambda : change_radiobutton_predict(4, patient_predict_combo, predict_button))
    rb5.pack(anchor="w")
    rb6=tk.Radiobutton(predict_frame, text="Movement (Real or Imagine) of Fists or Feets", variable=experience_predict_var, value=5, command = lambda : change_radiobutton_predict(5, patient_predict_combo, predict_button))
    rb6.pack(anchor="w")
    # rb1.select()
    # Framework for patient choices
    patient_frame_predict = tk.LabelFrame(onglet_predict, text="Patient")
    patient_frame_predict.pack(padx=10, pady=10)

    patient_predict_var = tk.StringVar()
    patients = get_predict(n_experience=experience_predict_var.get())

    patient_predict_combo = ttk.Combobox(patient_frame_predict, textvariable=patient_predict_var, values=patients, state="readonly")
    patient_predict_combo.set("Select patient")
    patient_predict_combo.pack(padx=10, pady=10)

    #button predict
    predict_button = tk.Button(onglet_predict, text="Predict", state="disabled", command=lambda:launch_process(patient=patient_predict_var.get(), experience=experience_predict_var.get(), type_process="PREDICT", drop_option=drop_option.get(), liste=patient_predict_combo['values'][1:]))
    predict_button.pack(padx=10, pady=10)

    #Interactiv
    patient_analyse_combo.bind("<<ComboboxSelected>>", lambda event:change_button_analyse(analys_button, patient_analyse_var.get()))
    patient_predict_combo.bind("<<ComboboxSelected>>", lambda event:change_button_predict(predict_button, patient_predict_var.get()))
    onglets.pack()

    return window