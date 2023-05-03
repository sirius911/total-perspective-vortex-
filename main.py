import numpy as numpy
import mne

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import tkinter as tk
from tkinter import ttk

from utils.experiments import experiments
from utils.utils_raw import get_raw, drop_bad_channels


def analyse(subject:int, n_experience:int):

    print("Process started with parameters : subject=", subject, ", experience=", n_experience)
    subject = int((subject))
    n_experience = int(n_experience)
    runs = experiments[n_experience]['runs']

    raw, events = get_raw(subject = subject, n_experience=n_experience, runs=runs)

    # draw the first data
    title = f"Patient #{subject} - {experiments[n_experience]['description'].title()} - BEFORE TRAITEMENT"
    
    raw.plot(scalings=dict(eeg=250e-6), title=title)
    plt.show()
    bad_channels = raw.info['bads']
    print(f"Bad_channels = {bad_channels}")
    raw = drop_bad_channels(raw, bad_channels)


    # drawing events
    event_dict = {value: key for key, value in experiments[n_experience]['mapping'].items()}
    fig = mne.viz.plot_events(events, sfreq = raw.info['sfreq'], first_samp=raw.first_samp, event_id=event_dict)
    fig.subplots_adjust(right= 0.8)

    # Perform spectral analysis on sensor data.
    raw.compute_psd(picks='all').plot()

    channels = raw.info["ch_names"]
    print(f"len (good_channels) = {len(channels)}")

    #ICA
    ica = mne.preprocessing.ICA(n_components=len(channels) - 2, random_state=0)
    raw_copy = raw.copy().filter(8,30)

    # The following electrodes have overlapping positions, which causes problems during visualization:
    raw_copy.drop_channels(['T9', 'T10'], on_missing='ignore')
    ica.fit(raw_copy)
    ica.plot_components(outlines='head', inst=raw_copy, show_names=False)

    print(f"Analyse of patient # {subject} ... done")

def change_button(event, patient):
    if patient == 'All':
        event['state'] = tk.DISABLED
    else:
        event['state'] = tk.NORMAL

def launch_process(patient, experience, type_process):
    if type_process == 'ANALYSE':
        analyse(patient, experience)

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
    for i in range(1, 110):
        patients.append(i)
    # patients = [str(i) for i in range(1,110)]

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

    # Buttons to start the process
    analys_button = tk.Button(window, text="Launch the analysis", state="disabled", command=lambda:launch_process(patient_var.get(), experience_var.get(), type_process='ANALYSE'))
    analys_button.pack(padx=10, pady=10)
    train_button = tk.Button(window, text="Train", state="active", command=lambda:launch_process(patient_var.get(), experience_var.get(), type_process='TRAIN'))
    train_button.pack(padx=0, pady=0)

    patient_combo.bind("<<ComboboxSelected>>", lambda event:change_button(analys_button, patient_var.get()))

    # Launching the event loop of the window
    window.mainloop()

if __name__ == "__main__":
    main_window()
    print("Good bye !")

