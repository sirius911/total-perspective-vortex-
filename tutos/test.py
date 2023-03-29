import os
import numpy as numpy
import mne
import matplotlib.pyplot as plt

#
files_name = mne.datasets.eegbci.load_data(subject=1, runs=[3, 7, 11],path=os.getenv('HOME') + '/goinfre')
print(files_name)

raw = mne.io.concatenate_raws([mne.io.read_raw_edf(f, preload=True) for f in files_name])
print(raw.info)
sfreq = raw.info['sfreq']
raw.plot(scalings=dict(eeg=250e-6), title='Before traitement')

events, event_id = mne.events_from_annotations(raw, event_id=dict(T0=0, T1=1, T2=2), verbose=True)
print(events.shape)
print(event_id.items())

annotations = mne.annotations_from_events(events=events, sfreq=sfreq, event_desc={0: "rest", 1: "left fist", 2:"right fist"}, verbose=True)
print(annotations)

fig = mne.viz.plot_events(events, sfreq=sfreq, first_samp=raw.first_samp, event_id=event_id)
fig.subplots_adjust(right= 0.7)
fig.title('Events')
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

#ICA
ica = mne.preprocessing.ICA(n_components=62, random_state=0)
dir(ica)
raw_copy = raw.copy().filter(8,30)
raw_copy.drop_channels(['T9', 'T10'])
ica.fit(raw_copy)
print(raw.ch_names)
ica.plot_components(outlines='head')

# Select channels
channels = raw.info["ch_names"]
good_channels = ["FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6",
                    "C5",  "C3",  "C1",  "Cz",  "C2",  "C4",  "C6",
                    "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6"]
bad_channels = [x for x in channels if x not in good_channels]

print(bad_channels)

raw.drop_channels(bad_channels)
raw.compute_psd().plot_topomap()

# Apply band-passfilter
raw.notch_filter(60, method="iir")
raw.compute_psd().plot()
plt.show()

raw.filter(7.0, 32.0, fir_design="firwin")
raw.compute_psd().plot()
plt.show()

raw.plot(scalings=dict(eeg=250e-6))
plt.show()