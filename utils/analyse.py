from .experiments import experiments
from .utils_raw import *
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def analyse(subject:int, n_experience:int, drop_option, options):
    print("Process started with parameters : subject=", subject, ", experience=", n_experience)
    subject = int((subject))
    n_experience = int(n_experience)
    runs = experiments[n_experience]['runs']
    name=get_name_model(subject=subject, n_experience=n_experience)
    raw, events = get_raw(subject = subject, n_experience=n_experience, runs=runs, drop_option=drop_option)

    # draw the first data
    title = f"Patient #{subject} - {experiments[n_experience]['description'].title()} - BEFORE TRAITEMENT"
    
    raw.plot(scalings=dict(eeg=250e-6), title=title)
    plt.show()

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
    if drop_option:
        raw = drop_bad_channels(raw=raw, name=name, save=True, verbose=True)
    print(f"Analyse of patient # {subject} ... done")