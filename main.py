import os
import numpy as numpy
import mne

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import tkinter as tk
from tkinter import ttk

path = os.getenv('HOME') + '/goinfre'

experiments = [
    {
        "description": "open and close left or right fist",
        "runs": [3, 7, 11],
        "mapping": {0: "rest", 1: "left fist", 2: "right fist"},
    },
    {
        "description": "imagine opening and closing left or right fist",
        "runs": [4, 8, 12],
        "mapping": {0: "rest", 1: "imagine left fist", 2: "imagine right fist"},
    },
    {
        "description": "open and close both fists or both feet",
        "runs": [5, 9, 13],
        "mapping": {0: "rest", 1: "both fists", 2: "both feets"},
    },
    {
        "description": "imagine opening and closing both fists or both feet",
        "runs": [6, 10, 14],
        "mapping": {0: "rest", 1: "imagine both fists", 2: "imagine both feets"},
    },
    {
        "description": "movement (real or imagine) of fists",
        "runs": [3, 7, 11, 4, 8, 12],
        "mapping": {0: "rest", 1: "left fist", 2: "right fist"},
    },
    {
        "description": "movement (real or imagine) of both fists or both feet",
        "runs": [5, 9, 13, 6, 10, 14],
        "mapping": {0: "rest", 1: "both fists", 2: "both feets"},
    },
]

def analyse(subject:int, n_experience:int):

    print("Process started with parameters : subject=", subject, ", experience=", n_experience)
    subject = int((subject))
    n_experience = int(n_experience)
    # n_experience = 0
    runs = experiments[n_experience]['runs']

    #load list of file for subject and #experience(runs)
    files_name = mne.datasets.eegbci.load_data(subject=subject, runs=runs ,path=path)
    # print(files_name)

    #concatenate all the file in one raw
    raw = mne.io.concatenate_raws([mne.io.read_raw_edf(f, preload=True, verbose=False) for f in files_name])
    print(raw.info)
    sfreq = raw.info['sfreq']

    # recup events and events_id
    events, event_id = mne.events_from_annotations(raw, event_id=dict(T0=0, T1=1, T2=2))
    # print(type(event_id))
    # print(events.shape)


    # set Descriptions of events in raw data
    annotations = mne.annotations_from_events(events=events, sfreq=sfreq, event_desc=experiments[n_experience]['mapping'], verbose=True)
    print(annotations)
    raw = raw.set_annotations(annotations=annotations)

    # Standardize channel positions and names.
    mne.datasets.eegbci.standardize(raw=raw)

    montage = mne.channels.make_standard_montage("biosemi64")
    raw.set_montage(montage, on_missing='ignore')


    # draw the first data
    title = f"Patient #{subject} - {experiments[n_experience]['description'].title()} - BEFORE TRAITEMENT"
    raw.plot(scalings=dict(eeg=250e-6), title=title)
    plt.show()

    # drawing events
    event_dict = {value: key for key, value in experiments[n_experience]['mapping'].items()}
    fig = mne.viz.plot_events(events, sfreq=sfreq, first_samp=raw.first_samp, event_id=event_dict)
    fig.subplots_adjust(right= 0.8)

    

    # Perform spectral analysis on sensor data.
    raw.compute_psd(picks='all').plot()

    #ICA
    ica = mne.preprocessing.ICA(n_components=62, random_state=0)
    raw_copy = raw.copy().filter(8,30)
    raw_copy.drop_channels(['T9', 'T10'])
    ica.fit(raw_copy)
    ica.plot_components(outlines='head', inst=raw, show_names=True)

    # Select channels
    channels = raw.info["ch_names"]
    good_channels = ["FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6",
                    "C5",  "C3",  "C1",  "Cz",  "C2",  "C4",  "C6",
                    "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6"]
    bad_channels = [x for x in channels if x not in good_channels]
    # print(bad_channels)
    raw.drop_channels(bad_channels)

    ###... in progress
    print(f"Analyse of patient # {subject} ... done")

def change_button(event):
    event['state'] = tk.NORMAL

def launch_process(patient, experience):
    analyse(patient, experience)

def main_window():
    # Create Main window
    window = tk.Tk()
    window.title("Physio / EEG")

    # Framework for patient choices
    patient_frame = tk.LabelFrame(window, text="Patient")
    patient_frame.pack(padx=10, pady=10)

    patient_var = tk.StringVar()
    patient_var.set("Choose a Patient")

    patients = [str(i) for i in range(1,110)]

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

    # Button to start the process
    launch_button = tk.Button(window, text="Launch the analysis", state="disabled", command=lambda:launch_process(patient_var.get(), experience_var.get()))
    launch_button.pack(padx=10, pady=10)

    patient_combo.bind("<<ComboboxSelected>>", lambda event:change_button(launch_button))

    # Launching the event loop of the window
    window.mainloop()

if __name__ == "__main__":
    main_window()
    print("Good bye !")

