import os
import mne
import json
from sklearn.model_selection import train_test_split
from .experiments import experiments, BAD_CHANNELS
from .commun import *
from joblib import dump, load



def get_path_bad_channels() -> str:
    return (f"{SAVE_PATH}/bad_channels/")

def load_bad_channels(name) -> list:
    path_bad_channels = f"{get_path_bad_channels()}{name}.json"
    with open(path_bad_channels, 'r') as file:
        bad_channels = json.load(file)
    return bad_channels

def save_bad_channels(bad_channels:list, name:str, verbose=False):
    path_bad_channels = f"{get_path_bad_channels()}{name}.json"
    list_to_save = bad_channels

    with open(path_bad_channels, 'w') as file:
        json.dump(list_to_save, file)
        if verbose:
            print(f"{colors.green} Saved{colors.reset}")

def drop_bad_channels(raw, name:str, save=False, verbose=False):
    """
        function deleting bad_channels to the raw
        if bad_channels == None it delete a predefined hard list 
    """
    bad_channels = raw.info['bads']
    raw.drop_channels(bad_channels)
    if verbose:
        print(f"{colors.red}Drop {len(bad_channels)} Bad channel(s).{colors.reset} -> ", end='')
    if save:
        save_bad_channels(bad_channels,name, verbose)
    return raw

def get_raw(subject, n_experience, drop_option):
    """
        function loading data from physionet
        args:
            subject: number of patient
            n_experience: number of experience
            runs: list of run

        return mne.raw and events

    """
    runs = experiments[n_experience]['runs']

    #load list of file for subject and #experience(runs)
    files_name = mne.datasets.eegbci.load_data(subject=subject, runs=runs ,path=PATH_DATA, verbose=50)

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

    # Drop bad_channels
    name=get_name_model(subject=subject, n_experience=n_experience)
    path_bad_channels = f"{get_path_bad_channels()}{name}.json"
    if drop_option:
        if os.path.exists(path_bad_channels):
            raw.info['bads'] = load_bad_channels(name)
        else:
            raw.info['bads'] = BAD_CHANNELS
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

def get_name_model(subject:int, n_experience:int) -> str:
    name = f"E{n_experience}S{subject:03d}"
    return name

def get_path_models() -> str:
    return (f"{SAVE_PATH}/models/")

def get_path(subject:int, n_experience:int):
    path = f"{get_path_models()}{get_name_model(subject, n_experience)}.mdl"
    return path

def my_filter(raw, verbose=False):
    # Apply band-pass filter
    raw.notch_filter(60, picks='eeg', method="iir", verbose = 50)
    raw.filter(5.0, 40.0, fir_design="firwin", skip_by_annotation="edge", verbose=50)
    return raw

def save_model(clf, path_model:str, verbose=False):
    dump(clf, path_model)
    if verbose:
        print(f"Model [{colors.blue}{path_model}{colors.reset}] was saved!")

def load_model(path_model):
    return load(path_model)

def exist(subject:int, n_experience:int) -> bool:
    return os.path.exists(get_path(subject= int(subject), n_experience= n_experience))

# def get_predict(n_experience:int):def perso_splitter(raw):
    """
    Return the data raw in 80% to train and 20% to test
    """
    X, Y = get_data(raw)
    X_train, X_test, y_train, y_test = train_test_split(X, Y,train_size=0.8, random_state=42)
    return X_train, X_test, y_train, y_test
#     """
#     return a list off subjects who can be predict
#     """
#     list_subject=[]
#     for subject in range(1,110):
#         if os.path.exist(subject=subject, n_experience=n_experience):
#             list_subject.append(subject)
#     if len(list_subject) > 1:
#         list_subject.insert(0, 'All')
#     return list_subject

def get_list_experience(subject:int) -> list:
    """
        return the list of experiences trained with the subject or [] if none
    """
    if subject == 'All':
        ensemble = set()
        for sub in range(1,110):
            for exp in range(6):
                if exist(subject= int(sub), n_experience= exp):
                    ensemble.update({exp})
            if len(ensemble) >=6:
                break
        return list(ensemble)
    else:
        list_exp = []
        for exp in range(6):
            if exist(subject= int(subject), n_experience= exp):
                    list_exp.append(exp)
        return list_exp

def what_predict(subject:int):
    """
       returns a list of experience numbers that can be predicted by the subject
    """
    ensemble = set()
    list_exp = get_list_experience(subject)
    if len(list_exp) > 0:
        for exp in list_exp:
            ensemble.update(experiments[exp]['predictions'])
    return list(ensemble)

def get_list_trained_subject():
    """
        returns the list of trained subjects or [] if none
    """
    list_subject = []
    for sub in range(110):
        for exp in range(6):
            if exist(subject=sub, n_experience=exp):
                list_subject.append(sub)
                break
    if len(list_subject) > 1:
        list_subject.insert(0, 'All')
    return list_subject

def perso_splitter(raw):
    """
    Return the data raw in 80% to train and 20% to test
    """
    X, Y = get_data(raw)
    X_train, X_test, y_train, y_test = train_test_split(X, Y,train_size=0.8, random_state=42)
    return X_train, X_test, y_train, y_test