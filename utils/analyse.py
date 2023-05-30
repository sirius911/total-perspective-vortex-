import os
from .experiments import experiments
from .utils_raw import *
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from .commun import get_json_value

def analyse(subject:int, n_experience:int, drop_option, options):
    print("Process started with parameters : subject=", subject, ", experience=", n_experience)
    subject = int((subject))
    n_experience = int(n_experience)
    name=get_name_model(subject=subject, n_experience=n_experience)
    # raw, events = get_raw(subject = subject, n_experience=n_experience, drop_option=drop_option)

    runs = experiments[n_experience]['runs']

    #load list of file for subject and #experience(runs)
    files_name = mne.datasets.eegbci.load_data(subject=subject, runs=runs ,path=get_json_value('PATH_DATA'), verbose=50)

    #concatenate all the file in one raw
    raw = mne.io.concatenate_raws([mne.io.read_raw_edf(f, preload=True, verbose=50) for f in files_name])
    
    sfreq = raw.info['sfreq']
    print(f"sfreq = {colors.blue}{sfreq}{colors.reset}")
    
    mne.datasets.eegbci.standardize(raw=raw)
    montage = mne.channels.make_standard_montage("biosemi64")
    raw.set_montage(montage, on_missing='ignore')
    path_bad_channels = f"{get_json_value('BAD_CHANNELS_DIR')}{name}.json"
    if drop_option and os.path.exists(path_bad_channels):
            raw.info['bads'] = load_bad_channels(name)
            print("ici")
            print(raw.info['bads'])
    # draw the first data
    title = f"Patient #{subject} - {experiments[n_experience]['description'].title()} - BEFORE TRAITEMENT"
    old_raw = raw.copy()
    old_raw.plot(scalings=dict(eeg=250e-6), title=title)
    plt.show()
    events, event_id = mne.events_from_annotations(raw, event_id=dict(T0=0, T1=1, T2=2))
    # annotations = mne.annotations_from_events(events=events, sfreq=sfreq, event_desc={0: "rest", 1: "left fist", 2:"right fist"}, verbose=True)
    if options.events.get():
        # drawing events
        event_dict = {value: key for key, value in experiments[n_experience]['mapping'].items()}
        fig = mne.viz.plot_events(events, sfreq = sfreq, first_samp=raw.first_samp, event_id=event_dict)
        fig.subplots_adjust(right= 0.8)
    raw.filter(7.0, 30.0, fir_design="firwin", skip_by_annotation="edge")
    # montage = mne.channels.make_standard_montage("biosemi64")
    # raw.set_montage(montage, on_missing='ignore')
    picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False)
    print(len(picks))
    n_components = 20
    if len(picks) <= 18:
         n_components = len(picks) - 2
    if n_components >2:
        ica = mne.preprocessing.ICA(n_components=n_components, random_state=0, method='infomax')
        raw.drop_channels(['T9', 'T10'], on_missing='ignore')
        ica.fit(raw.copy())
        l = set()
        bad_ = set()
        for channel in raw.info["ch_names"]:
            bad_idx, scores = ica.find_bads_eog(raw, ch_name=channel, threshold=2, verbose=False)
            bad_.add(channel)
            for i in bad_idx:
                l.add(i)
        print(list(l))
        print(list(bad_))
        ica.exclude = list(l)
        ica.apply(raw.copy(), exclude=ica.exclude)
    if options.ica.get():
        ica.plot_components(outlines='head', inst=raw)
        picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False)
        epochs = mne.Epochs(raw, events, event_id, -1.0, 4.0,proj=True, picks=picks, baseline=None, preload=True, verbose=50)
        epochs = ica.apply(epochs, exclude=ica.exclude)
        epochs.apply_baseline((None, 0))
        epochs.equalize_event_counts(event_id)
        epochs.plot(title="Epochs")
    # raw.info['bads'] = list(bad_)
    if options.spectral.get():
        # Perform spectral analysis on sensor data.
        raw.compute_psd(picks='all').plot(picks="data", exclude="bads")

    # if options.ica.get():
    #     # channels = raw.info["ch_names"]

    #     raw_copy = my_filter(raw.copy())
    #     # raw_copy = drop_bad_channels(raw=raw_copy, name=name, save=False, verbose=False)
    #     # The following electrodes have overlapping positions, which causes problems during visualization:
    #     # raw_copy.drop_channels(['T9', 'T10'], on_missing='ignore')
    #     #ICA
    #     ica = mne.preprocessing.ICA(n_components=20, random_state=0)
    #     ica.fit(raw_copy)
    #     # Identification des ICs liées aux mouvements oculaires
    #     # eog_epochs = mne.preprocessing.create_eog_epochs(raw=raw_copy, ch_name=raw_copy.info["ch_names"], picks='eeg')
    #     eog_evoked = mne.preprocessing.create_eog_epochs(raw_copy, ch_name=raw.info['ch_names']).average(picks="all")
    #     # blinks
    #     ica.plot_overlay(raw, exclude=[0], picks="eeg")
    #     ica.exclude = []
    #     eog_indices, eog_scores = ica.find_bads_eog(raw_copy, ch_name="Fpz")
    #     # Affichage des résultats
    #     ica.plot_scores(eog_scores, exclude=eog_indices)  # Visualisation des scores des ICs
    #     ica.plot_sources(eog_evoked)  # Visualisation des ICs dans les données brutes
    #     # Application des modifications aux données
        
    # draw the after traitement data
    title = f"Patient #{subject} - {experiments[n_experience]['description'].title()} - AFTER TRAITEMENT"
    raw = my_filter(raw)
    raw.plot(title=title)
    plt.show()
    if drop_option:
        raw = drop_bad_channels(raw=raw, name=name, save=True, verbose=True)
    print(f"Analyse of patient # {subject} ... done")