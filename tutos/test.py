import os
import numpy as numpy
import mne
import matplotlib.pyplot as plt

#
files_name = mne.datasets.eegbci.load_data(subject=1, runs=[3, 7, 11],path=os.getenv('HOME') + '/goinfre')
print(files_name)

raw = mne.io.concatenate_raws([mne.io.read_raw_edf(f, preload=True) for f in files_name])
sfreq = raw.info['sfreq']
print(f"sfreq = {sfreq}")

events, event_id = mne.events_from_annotations(raw, event_id=dict(T0=0, T1=1, T2=2), verbose=True)
print(events.shape)
print(event_id.items())

annotations = mne.annotations_from_events(events=events, sfreq=sfreq, event_desc={0: "rest", 1: "left fist", 2:"right fist"}, verbose=True)
print(annotations)

fig = mne.viz.plot_events(events, sfreq=sfreq, first_samp=raw.first_samp, event_id=event_id)
fig.subplots_adjust(right= 0.7)
plt.show()

raw = raw.set_annotations(annotations=annotations, verbose = True)

mne.datasets.eegbci.standardize(raw=raw)

montage = mne.channels.make_standard_montage("biosemi64")
montage.plot()
raw.set_montage(montage, on_missing='ignore')
raw.plot(scalings=dict(eeg=250e-6)) #range 500µV


plt.show()

raw.compute_psd(picks='all').plot()
raw.compute_psd(picks='all').plot(average=True)

plt.show()