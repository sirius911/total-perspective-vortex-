import tkinter as tk
from tkinter import ttk

from .utils_raw import get_list_trained_subject, get_list_experience
from .utils_csp import load_algo
from .run import launch_process
from .experiments import experiments
from .commun import *

def click_predict_choice(objet):
    button = objet.window.predict_button
    if objet.window.trained_choice.has_select():
        button['state'] = tk.NORMAL
    else:
        button['state'] = tk.DISABLED


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

class Predict_choice:
    def __init__(self,window, parent):
        self.window = window
        self.box = []
        self.val = tk.IntVar(value=0)
        for num, exp in enumerate(experiments):
            self.box.append(tk.Radiobutton(parent, text=exp["description"], variable=self.val, value = num, state="disabled",  command=lambda:click_predict_choice(self)))
        for b in self.box:
            b.pack(anchor="w")

    def enable(self, row_index=-1):
        if row_index == -1:
            for b in self.box:
                b.configure(state="active")
        else:
            self.box[row_index].configure(state="active")
    
    def disabled(self, row_index=-1):
        if row_index == -1:
            for b in self.box:
                b.configure(state="disabled")
        else:
            self.box[row_index].configure(state="disabled")
    
    def get_exp(self):
        """
            return the number of experience
        """
        return self.val.get()

    def has_select(self):
        return self.val != 0

def change_button_analyse(analys_button, patient):
    if patient == 'All':
        analys_button['state'] = tk.DISABLED
    else:
        analys_button['state'] = tk.NORMAL
    

def change_button_predict(patient, window):

    trained_choice = window.trained_choice

    with_wath = get_list_experience(patient)
    for exp in range(6):
        if exp in with_wath:
            trained_choice.enable(exp)
        else:
            trained_choice.disabled(exp)
    window.update()    

def reload_predict_tab(patient_predict_var, patient_predict_combo):
    new_patients = get_list_trained_subject()
    patient_predict_var.set("Select patient")
    patient_predict_combo["values"] = new_patients
    if new_patients:
        patient_predict_combo.set(new_patients[0])

def save_changes(root, entry:list):
    data = {
    "SAVE_PATH": entry[0].get(),
    "PATH_DATA": entry[1].get(),
    "MODELS_PATH_DIR": entry[2].get(),
    "BAD_CHANNELS_DIR": entry[3].get(),
    "ALGO_PATH": entry[4].get(),
    }
    with open("utils/path.json", 'w') as file:
        json.dump(data, file, indent=4)
    SAVE_PATH = get_json_value("SAVE_PATH")
    PATH_DATA = get_json_value("PATH_DATA")
    MODELS_PATH_DIR = get_json_value("MODELS_PATH_DIR")
    BAD_CHANNELS_DIR = get_json_value("BAD_CHANNELS_DIR")
    ALGO_PATH = get_json_value("ALGO_PATH") 
    root.destroy()

def open_paths_window(root):
    labels = []
    entrys = []
    paths_window = tk.Toplevel(root)
    paths_window.title("Folders")
    paths_window.geometry("500x350")
    data = get_data_path()
    for label, entry in data.items():
        lab = tk.Label(paths_window, text=label)
        lab.pack()
        labels.append(lab)
        ent = tk.Entry(paths_window)
        ent.insert(0, entry)
        ent.pack(fill=tk.X, padx=10, pady=5)
        entrys.append(ent)

    save_button = tk.Button(paths_window, text="Save", command= lambda:save_changes(paths_window, entrys))
    save_button.pack(padx=10, pady=10)


def quit_program(root):
        root.quit()

def create_window(window) -> tk:
    window.title("PhysioNet / EEG")

    # Crée le menu principal
    menu_bar = tk.Menu(window)
    window.config(menu=menu_bar)

    # Crée le menu "File"
    file_menu = tk.Menu(menu_bar, tearoff=False)
    menu_bar.add_cascade(label="File", menu=file_menu)
    file_menu.add_command(label="Quit", command=lambda: quit_program(window))

    # Crée le menu "Paths"
    paths_menu = tk.Menu(menu_bar, tearoff=False)
    menu_bar.add_cascade(label="Folders", menu=paths_menu)
    paths_menu.add_command(label="Open Folders Window", command=lambda: open_paths_window(window))

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
    drop_option_analyse =  tk.BooleanVar(value=True)
    drop_checkbutton_analyse = tk.Checkbutton(onglet_analyse, text="Drop Bad Channels", variable=drop_option_analyse)
    drop_checkbutton_analyse.pack(padx=10, pady=10)

    #frames
    options_analyse_frame = tk.LabelFrame(onglet_analyse, text="Options")
    options_analyse_frame.pack(padx=10, pady=10)

    options = Option(options_analyse_frame)

    # button Analyse
    analys_button = tk.Button(onglet_analyse, text="Launch the analysis", state="disabled", command=lambda:launch_process(patient_analyse_var.get(), experience_var.get(), type_process='ANALYSE', drop_option=drop_option_analyse.get(), options=options))
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
    
    # Checkbox for Drop bad channels option
    drop_option_predict =  tk.BooleanVar(value=True)
    drop_checkbutton_predict = tk.Checkbutton(onglet_training, text="Drop Bad Channels", variable=drop_option_predict)
    drop_checkbutton_predict.pack(padx=10, pady=10)

    #CSP
    csp_var = tk.StringVar()
    csp_var.set("set CSP")
    csp = ['mne.decoding.CSP']
    csp.extend(load_algo())
    csp_combo = ttk.Combobox(onglet_training, textvariable=csp_var, values = csp, state="readonly")
    csp_combo.current(csp.index('mne.decoding.CSP'))
    csp_combo.pack(padx=10, pady=10)

    #button train
    train_button = tk.Button(onglet_training, text="Train", command=lambda:launch_process(patient_train_var.get(), experience_train_var.get(), type_process='TRAIN', drop_option=drop_option_predict.get(), options=csp_var.get()))
    train_button.pack(padx=10, pady=10)

    ###### ONGLET PREDICT #######
    # Framework for patient choices
    patient_frame_predict = tk.LabelFrame(onglet_predict, text="Patient")
    patient_frame_predict.pack(padx=10, pady=10)

    patient_predict_var = tk.StringVar()
    patients = get_list_trained_subject()

    patient_predict_combo = ttk.Combobox(patient_frame_predict, textvariable=patient_predict_var, values=patients, state="readonly")
    patient_predict_combo.set("Select patient")
    patient_predict_combo.pack(padx=10, pady=10)

    predict_trained_frame = tk.LabelFrame(onglet_predict, text="Select trained model ")
    predict_trained_frame.pack(padx=2, pady=2)
    window.trained_choice = Predict_choice(window, predict_trained_frame)

    #button predict
    window.predict_button = tk.Button(onglet_predict, text="Predict", state="disabled", command=lambda:launch_process(patient=patient_predict_var.get(), experience=window.trained_choice.get_exp(), type_process='PREDICT'))
    
    window.predict_button.pack(padx=10, pady=10)

    #Interactiv
    patient_analyse_combo.bind("<<ComboboxSelected>>", lambda event:change_button_analyse(analys_button, patient_analyse_var.get()))
    patient_predict_combo.bind("<<ComboboxSelected>>", lambda event:change_button_predict(patient_predict_var.get(), window))
    onglets.bind("<<NotebookTabChanged>>",lambda event:reload_predict_tab(patient_predict_var, patient_predict_combo))
    onglets.pack()

    return window