import os
import mne
from .experiments import experiments
from .commun import colors

path = os.getenv('HOME') + '/goinfre'

def drop_bad_channels(raw, bad_channels=None, verbose=False):
    """
        functoin deleting bad_channels to the raw
        if bad_channels == None it delete a predefined hard list 
    """
    channels = raw.info["ch_names"]
    if len(bad_channels) == 0:
        # good_channels = ["FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6",
        #                 "C5",  "C3",  "C1",  "Cz",  "C2",  "C4",  "C6",
        #                 "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6"]
        
        # bad_channels = [x for x in channels if x not in good_channels]
        bad_channels = ['PO7', 'P8', 'P6', 'P2', 'P1', 'P5', 'P7', 'TP8', 'TP7', 'T10', 'T9', 'T8', 'T7', 'FT8', 'FT7', 'F2', 'F8', 'F1', 'F5', 'F7', 'AF8', 'AF7', 'Fp2', 'Fpz', 'CP6', 'CP2', 'CP1', 'CP5', 'C6', 'C2', 'C5', 'FC6', 'FC2', 'FC5', 'FC1', 'PO8', 'O1', 'O2', 'F6']
    raw.drop_channels(bad_channels)
    if verbose:
        print(f"{colors.red}Drop {len(bad_channels)} Bad channel(s).{colors.reset}")
    return raw

def get_raw(subject, n_experience, runs):
    """
        function loading data from physionet
        args:
            subject: number of patient
            n_experience: number of experience
            runs: list of run

        return mne.raw and events

    """
    #load list of file for subject and #experience(runs)
    files_name = mne.datasets.eegbci.load_data(subject=subject, runs=runs ,path=path)

    #concatenate all the file in one raw
    raw = mne.io.concatenate_raws([mne.io.read_raw_edf(f, preload=True, verbose=50) for f in files_name])
    sfreq = raw.info['sfreq']

    # recup events and events_id
    events, event_id = mne.events_from_annotations(raw, event_id=dict(T0=0, T1=1, T2=2), verbose=50)

    # set Descriptions of events in raw data
    annotations = mne.annotations_from_events(events=events, sfreq=sfreq, event_desc=experiments[n_experience]['mapping'], verbose=50)
    raw = raw.set_annotations(annotations=annotations, verbose=50)

    # Standardize channel positions and names.
    mne.datasets.eegbci.standardize(raw=raw)

    # Montage
    montage = mne.channels.make_standard_montage("biosemi64")
    raw.set_montage(montage, on_missing='ignore')

    return raw, events

def get_data(raw):
    """
        function return Epoch and labels
    """
    tmin, tmax = -1.0, 4.0
    events, event_id = mne.events_from_annotations(raw, verbose=50)
    picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False)
    epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks, baseline=None, preload=True, verbose=50)
    labels = epochs.events[:, -1]
    # print(labels)
    epochs_train = epochs.copy().crop(tmin=1.0, tmax=4.0).get_data()
    # print(epochs_train.shape)
    return epochs_train, labels

def my_filter(raw, verbose=False):
    # Apply band-pass filter
    # raw.notch_filter(60, picks='eeg', method="iir", verbose = 50)
    raw.filter(5.0, 40.0, fir_design="firwin", skip_by_annotation="edge", verbose=50)
    return raw