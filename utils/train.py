import sys
import matplotlib.pyplot as plt
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.metrics import accuracy_score
from mne.decoding import CSP 
from .csp import CSP as MY_CSP
from mne.preprocessing import ICA


from .experiments import experiments
from .utils_raw import *
from .commun import colorize
from .graph import plot_learning_curve

def train2(subject:int, n_experience:int, drop_option, verbose=False):
    if verbose:
        print("Process start with parameters : subject=", subject, ", experience=", n_experience)
    n_experience = int(n_experience)
    raw, events, event_id = get_raw(subject = subject, n_experience=n_experience, drop_option=drop_option)
    
    if drop_option:
        raw = drop_bad_channels(raw=raw, name=get_name_model(subject=subject, n_experience=n_experience),save=False , verbose=verbose)
    
    raw = my_filter(raw)
    
    ica = ICA(n_components=20, random_state=0)
    ica.fit(raw)
    # Exclure les composantes ICA indésirables
    ica.exclude = [2, 3, 15, 16, 17, 18, 19]
    cleaned_raw = raw.copy()    
    ica.apply(cleaned_raw, exclude=ica.exclude)
    picks = mne.pick_types(cleaned_raw.info, meg=False, eeg=True, stim=False, eog=False)
    epochs = mne.Epochs(cleaned_raw, events, event_id, -1, 4, proj=True, picks=picks, baseline=None, preload=True, verbose=50)
    epochs = ica.apply(epochs, exclude=ica.exclude)
    epochs.apply_baseline((None, 0))
    epochs.equalize_event_counts(event_id)
    epochs.pick_types(eeg=True)

    X_train, _, y_train, _ = perso_splitter(epochs)
    input("ici")
    # Assemble a classifier #2
    csp = CSP(n_components=6, log=True,norm_trace=False)
    lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    clf= Pipeline([("ica", ica), ("CSP", csp), ("LDA", lda)], verbose=False)
    

    # Rediriger la sortie vers null
    default_stdout = sys.stdout
    # sys.stdout = open('/dev/null', 'w')

    clf.fit(X_train, y_train)
    cvs = cross_val_score(clf, X_train, y_train)
    mean_cvs = np.mean(cvs)

    # Restaurer la sortie par défaut
    # sys.stdout = default_stdout

    if verbose:
        print(f"cvs = {cvs}")
        print(f"mean of Cross_Val_Score =  = {colorize(mean_cvs)}")
        # title = "Learning Curves "
        # plot_learning_curve(clf, title, X_train, y_train, n_jobs=-1)
        print(f"Training with Patient #{colors.blue}{subject}{colors.reset} [{colors.green}{experiments[n_experience]['description']}{colors.reset}] ... Done")
    #Save
    clf.drop_option = drop_option
    save_model(clf, get_path(subject, n_experience), verbose=verbose)
    return mean_cvs

def train(subject:int, n_experience:int, drop_option, csp = "mne.CSP", verbose=False):
    if verbose:
        print("Process start with parameters : subject=", subject, ", experience=", n_experience)
    n_experience = int(n_experience)
    raw, _, _ = get_raw(subject = subject, n_experience=n_experience, drop_option=drop_option)
    
    if drop_option:
        raw = drop_bad_channels(raw=raw, name=get_name_model(subject=subject, n_experience=n_experience),save=False , verbose=verbose)
    
    raw = my_filter(raw)
    
    X_train, _, y_train, _ = perso_splitter(raw)
    
    # Assemble a classifier #2
    if csp== "mne.CSP":
        csp = CSP(n_components=6, log=True,norm_trace=False)
    else:
        csp = MY_CSP(n_components=6)
    lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    clf= Pipeline([("CSP", csp), ("LDA", lda)], verbose=False)
    

    # Rediriger la sortie vers null
    default_stdout = sys.stdout
    sys.stdout = open('/dev/null', 'w')

    clf.fit(X_train, y_train)
    cvs = cross_val_score(clf, X_train, y_train)
    mean_cvs = np.mean(cvs)

    # Restaurer la sortie par défaut
    sys.stdout = default_stdout

    if verbose:
        print(f"cvs = {cvs}")
        print(f"mean of Cross_Val_Score =  = {colorize(mean_cvs)}")
        # title = "Learning Curves "
        # plot_learning_curve(clf, title, X_train, y_train, n_jobs=-1)
        print(f"Training with Patient #{colors.blue}{subject}{colors.reset} [{colors.green}{experiments[n_experience]['description']}{colors.reset}] ... Done")
    #Save
    clf.drop_option = drop_option
    save_model(clf, get_path(subject, n_experience), verbose=verbose)
    return mean_cvs
