import os
import mne
from .experiments import experiments

path = os.getenv('HOME') + '/goinfre'

def get_raw(subject, n_experience, runs):
    """
        function loading data from physionet

    """
    #load list of file for subject and #experience(runs)
    files_name = mne.datasets.eegbci.load_data(subject=subject, runs=runs ,path=path)

    #concatenate all the file in one raw
    raw = mne.io.concatenate_raws([mne.io.read_raw_edf(f, preload=True, verbose=False) for f in files_name])
    sfreq = raw.info['sfreq']

    # recup events and events_id
    events, event_id = mne.events_from_annotations(raw, event_id=dict(T0=0, T1=1, T2=2))

    # set Descriptions of events in raw data
    annotations = mne.annotations_from_events(events=events, sfreq=sfreq, event_desc=experiments[n_experience]['mapping'], verbose=True)
    raw = raw.set_annotations(annotations=annotations)

    # Standardize channel positions and names.
    mne.datasets.eegbci.standardize(raw=raw)

    # Montage
    montage = mne.channels.make_standard_montage("biosemi64")
    raw.set_montage(montage, on_missing='ignore')

    return raw, events
