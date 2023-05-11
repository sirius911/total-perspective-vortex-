import sys
import numpy as np
import mne

from mne import Epochs
from mne.decoding import CSP 

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import tkinter as tk
from tkinter import ttk

from utils.experiments import experiments
from utils.utils_raw import get_raw, drop_bad_channels, get_data, my_filter
from utils.commun import colors
from utils.graph import plot_learning_curve

from utils.utils_window import Option, change_button

from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, train_test_split, ShuffleSplit
from sklearn.metrics import accuracy_score
from sklearn.utils import parallel_backend
from tqdm import tqdm

model = {
    'train':False,
    'patient_list':[],
    'clf':None,
    'saved':False,
}

def valid(test):
    result=""
    if test:
        result = (f"{colors.green}Ok")
    else:
        result = (f"{colors.red}Ko")
    result += (f"{colors.reset}")
    return result

def train(subject:int, n_experience:int, drop_option, model, verbose=False):
    if verbose:
        print("Process started with parameters : subject=", subject, ", experience=", n_experience)
    n_experience = int(n_experience)
    runs = experiments[n_experience]['runs']
    raw, _ = get_raw(subject = subject, n_experience=n_experience, runs=runs)
    
    if drop_option:
        bad_channels = raw.info['bads']
        raw = drop_bad_channels(raw, bad_channels, verbose)
    
    raw = my_filter(raw, verbose)
    # Read epochs (events)
    epochs_train, labels = get_data(raw)
    
    cv = ShuffleSplit(10, test_size=0.2, random_state=0)
    if model['clf'] == None:
        # Assemble a classifier #2
        csp = CSP(2)
        lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
        model['clf'] = Pipeline([("CSP", csp), ("LDA", lda)], verbose=False)
    
    # fit our pipeline to the experiment #1
    _, X_test, _, y_test = train_test_split(epochs_train, labels, random_state=0)
    if verbose == False:
        default_stdout = sys.stdout
        # Rediriger la sortie vers null
        sys.stdout = open('/dev/null', 'w')

    model['clf'].fit(epochs_train, labels)
    predictions = model['clf'].predict(X_test)
    score = accuracy_score(predictions, y_test)
    if verbose == False:
        # Restaurer la sortie par défaut
        sys.stdout = default_stdout
    else:
        title = "Learning Curves "
        plot_learning_curve(model['clf'], title, epochs_train, labels,cv=cv, n_jobs=-1)
        plt.show()
        print(f"Training with Patient #{colors.blue}{subject}{colors.reset} [{colors.green}{experiments[n_experience]['description']}{colors.reset}] ... Done")
    model['train'] = True
    model['patient_list'].append(subject)
    return model,score

def predict(subject:int, n_experience:int, drop_option, model, verbose = False):
    print("Process start with parameters : subject=", subject, ", experience=", n_experience)
    
    subject = int((subject))
    n_experience = int(n_experience)
    runs = experiments[n_experience]['runs']
    raw, _ = get_raw(subject = subject, n_experience=n_experience, runs=runs)
    if drop_option:
        bad_channels = raw.info['bads']
        raw = drop_bad_channels(raw, bad_channels, verbose)

    # Apply band-pass filter
    raw = my_filter(raw,verbose=verbose)

    cv = ShuffleSplit(10, test_size=0.2, random_state=0)

    # Read epochs
    epochs_train, labels = get_data(raw)
    _, X_test, _, y_test = train_test_split(epochs_train, labels, random_state=0)
    if verbose == False:
        default_stdout = sys.stdout
        # Rediriger la sortie vers null
        sys.stdout = open('/dev/null', 'w')
    predictions = model['clf'].predict(X_test)
    scores_ldashrinkage = cross_val_score(model['clf'], epochs_train, labels, cv=cv, n_jobs=-1, verbose=0)
    mean_scores_ldashrinkage = np.mean(scores_ldashrinkage)

    if verbose == False:
        # Restaurer la sortie par défaut
        sys.stdout = default_stdout
    else:
        print(f'event nb: [prediction] [truth] equal?')
        for i, prediction in enumerate(predictions):
            print(f'event {i:02d}: [{prediction}] [{y_test[i]}] {valid(prediction == y_test[i])}')
        score_subject = accuracy_score(predictions, y_test)
        print(f'Mean accuracy for all experiments:{score_subject}')
        print(f"Mean cross val score {mean_scores_ldashrinkage}")
    return mean_scores_ldashrinkage

def analyse(subject:int, n_experience:int, drop_option, options):

    print("Process started with parameters : subject=", subject, ", experience=", n_experience)
    subject = int((subject))
    n_experience = int(n_experience)
    runs = experiments[n_experience]['runs']

    raw, events = get_raw(subject = subject, n_experience=n_experience, runs=runs)

    # draw the first data
    title = f"Patient #{subject} - {experiments[n_experience]['description'].title()} - BEFORE TRAITEMENT"
    
    raw.plot(scalings=dict(eeg=250e-6), title=title)
    plt.show()

    if drop_option:
        bad_channels = raw.info['bads']
        raw = drop_bad_channels(raw, bad_channels, verbose=True)

    if options.events.get():
        # drawing events
        event_dict = {value: key for key, value in experiments[n_experience]['mapping'].items()}
        fig = mne.viz.plot_events(events, sfreq = raw.info['sfreq'], first_samp=raw.first_samp, event_id=event_dict)
        fig.subplots_adjust(right= 0.8)

    if options.spectral.get():
        # Perform spectral analysis on sensor data.
        raw.compute_psd(picks='all').plot()

    if options.ica.get():
        channels = raw.info["ch_names"]

        #ICA
        ica = mne.preprocessing.ICA(n_components=len(channels) - 2, random_state=0)
        raw_copy = raw.copy().filter(8,30)

        # The following electrodes have overlapping positions, which causes problems during visualization:
        raw_copy.drop_channels(['T9', 'T10'], on_missing='ignore')
        ica.fit(raw_copy)
        ica.plot_components(outlines='head', inst=raw_copy, show_names=False)

    # draw the after traitement data
    title = f"Patient #{subject} - {experiments[n_experience]['description'].title()} - AFTER TRAITEMENT"
    raw = my_filter(raw, verbose=True)
    raw.plot(scalings=dict(eeg=250e-6), title=title)
    plt.show()
    print(f"Analyse of patient # {subject} ... done")



def launch_process(patient, experience, type_process, drop_option=True, options=None):
    global model
    verbose = False
    if patient == 'All':
        subjects = range(1, 5)
    else:
        subjects = range(int(patient), int(patient) + 1)
        verbose = True
    if type_process == 'ANALYSE':
        analyse(patient, experience, drop_option, options=options)
    elif type_process == 'TRAIN':
        scores =[]
        for subject in tqdm(subjects):
            model, score = train(subject, experience, drop_option, model=model, verbose=verbose)
            scores.append(score)
        print(f"Training ...{colors.green}Ok{colors.reset}")
        print(f"Model['train'] = {model['train']}")
        print(f"Mean score = {np.mean(scores)}")
    elif type_process == "PREDICT":
        score = []
        for subject in tqdm(subjects):
            score.append(predict(subject, experience, drop_option, model=model, verbose=verbose))
        print (f"mean = {np.mean(score)}")
    return model



def main_window():
    # Create Main window
    window = tk.Tk()
    window.title("PhysioNet / EEG")

    # Framework for patient choices
    patient_frame = tk.LabelFrame(window, text="Patient")
    patient_frame.pack(padx=10, pady=10)

    patient_var = tk.StringVar()
    patient_var.set("Set Patient")
    patients = []
    patients.append("All")
    patients.extend(range(1, 110))

    patient_combo = ttk.Combobox(patient_frame, textvariable=patient_var, values=patients, state="readonly")
    patient_combo.pack(padx=10, pady=10)

    # Framework for experience choices
    experience_frame = tk.LabelFrame(window, text="Experience")
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
    drop_checkbutton = tk.Checkbutton(window, text="Drop Bad Channels", variable=drop_option)
    drop_checkbutton.pack(padx=10, pady=10)

    #frames
    analyse_frame = tk.LabelFrame(window, text="Analyse")
    analyse_frame.pack(padx=10, pady=10)

    train_frame = tk.LabelFrame(window, text="Training")
    train_frame.pack(padx=2, pady=2)
    
    options = Option(analyse_frame)
    # Buttons to start a process
    # button Analyse
    analys_button = tk.Button(analyse_frame, text="Launch the analysis", state="disabled", command=lambda:launch_process(patient_var.get(), experience_var.get(), type_process='ANALYSE', drop_option=drop_option.get(), options=options))
    analys_button.pack(padx=10, pady=10)
    #button train
    train_button = tk.Button(train_frame, text="Train", state="disabled", command=lambda:launch_process(patient_var.get(), experience_var.get(), type_process='TRAIN', drop_option=drop_option.get()))
    train_button.pack(padx=10, pady=10)
    #button predict
    predict_button = tk.Button(train_frame, text="Predict", command=lambda:launch_process(patient=patient_var.get(), experience=experience_var.get(), type_process="PREDICT", drop_option=drop_option.get()))
    predict_button.pack(padx=10, pady=10)
    patient_combo.bind("<<ComboboxSelected>>", lambda event:change_button(analys_button, train_button, patient_var.get()))

    # Launching the event loop of the window
    window.mainloop()

if __name__ == "__main__":
    main_window()
    print("Good bye !")

