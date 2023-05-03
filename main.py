import numpy as np
import mne
from mne import Epochs, pick_types, annotations_from_events, events_from_annotations
from mne.decoding import CSP 

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import tkinter as tk
from tkinter import ttk

from utils.experiments import experiments
from utils.utils_raw import get_raw, drop_bad_channels

from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import learning_curve, cross_val_score, train_test_split

def train(subject:int, n_experience:int, drop_option):
    print("Process started with parameters : subject=", subject, ", experience=", n_experience)
    tmin, tmax = -1.0, 4.0
    subject = int((subject))
    n_experience = int(n_experience)
    runs = experiments[n_experience]['runs']
    raw, events = get_raw(subject = subject, n_experience=n_experience, runs=runs)
    if drop_option:
        bad_channels = raw.info['bads']
        raw = drop_bad_channels(raw, bad_channels)

    # Apply band-pass filter
    raw.notch_filter(60, method="iir")
    raw.filter(7.0, 32.0, fir_design="firwin", skip_by_annotation="edge")

    # Read epochs
    events, event_id = mne.events_from_annotations(raw)
    picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False)
    epochs = Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks, baseline=None, preload=True)
    labels = epochs.events[:, -1]
    # print(labels)
    epochs_train = epochs.copy().crop(tmin=1.0, tmax=4.0).get_data()
    print(epochs_train.shape)

    # Assemble a classifier
    csp = CSP(6)
    lda = LinearDiscriminantAnalysis()
    clf = Pipeline([("CSP", csp), ("LDA", lda)])

    # fit our pipeline to the experiment
    X_train, X_test, y_train, y_test = train_test_split(epochs_train, labels, random_state=0)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    train_sizes, train_scores, test_scores = learning_curve(clf, epochs_train, labels, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10), scoring='accuracy')
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)

    plt.plot(train_sizes, train_scores_mean, label='Training score')
    plt.plot(train_sizes, test_scores_mean, label='Cross-validation score')
    plt.xlabel('Number of Events')
    plt.ylabel('Accuracy score')
    plt.legend()
    titre = (f"Patient #{subject} [{experiments[n_experience]['description']}] Score ={score:0.02f}")
    plt.title(titre, fontsize=9)
    plt.show()

def analyse(subject:int, n_experience:int, drop_option):

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
        raw = drop_bad_channels(raw, bad_channels)


    # drawing events
    event_dict = {value: key for key, value in experiments[n_experience]['mapping'].items()}
    fig = mne.viz.plot_events(events, sfreq = raw.info['sfreq'], first_samp=raw.first_samp, event_id=event_dict)
    fig.subplots_adjust(right= 0.8)

    # Perform spectral analysis on sensor data.
    raw.compute_psd(picks='all').plot()

    channels = raw.info["ch_names"]

    #ICA
    ica = mne.preprocessing.ICA(n_components=len(channels) - 2, random_state=0)
    raw_copy = raw.copy().filter(8,30)

    # The following electrodes have overlapping positions, which causes problems during visualization:
    raw_copy.drop_channels(['T9', 'T10'], on_missing='ignore')
    ica.fit(raw_copy)
    ica.plot_components(outlines='head', inst=raw_copy, show_names=False)

    print(f"Analyse of patient # {subject} ... done")

def change_button(analys_button, train_button, patient):
    if patient == 'All':
        analys_button['state'] = tk.DISABLED
    else:
        analys_button['state'] = tk.NORMAL
    train_button['state'] = tk.NORMAL

def launch_process(patient, experience, type_process, drop_option=True):
    if type_process == 'ANALYSE':
        analyse(patient, experience, drop_option)
    elif type_process == 'TRAIN':
        train(patient, experience, drop_option)

def main_window():
    # Create Main window
    window = tk.Tk()
    window.title("Physio / EEG")

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
    drop_option =  tk.BooleanVar(value=False)
    drop_checkbutton = tk.Checkbutton(window, text="Drop Bad Channels", variable=drop_option)
    drop_checkbutton.pack(padx=10, pady=10)

    # Buttons to start the process
    analys_button = tk.Button(window, text="Launch the analysis", state="disabled", command=lambda:launch_process(patient_var.get(), experience_var.get(), type_process='ANALYSE',drop_option=drop_option.get()))
    analys_button.pack(padx=10, pady=10)
    train_button = tk.Button(window, text="Train", state="disabled", command=lambda:launch_process(patient_var.get(), experience_var.get(), type_process='TRAIN',drop_option=drop_option.get()))
    train_button.pack(padx=0, pady=0)

    patient_combo.bind("<<ComboboxSelected>>", lambda event:change_button(analys_button, train_button, patient_var.get()))

    # Launching the event loop of the window
    window.mainloop()

if __name__ == "__main__":
    main_window()
    print("Good bye !")

