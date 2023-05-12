import sys
import matplotlib.pyplot as plt
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.metrics import accuracy_score
from mne.decoding import CSP 

from .experiments import experiments
from .utils_raw import *
from .graph import plot_learning_curve

def train(subject:int, n_experience:int, drop_option, verbose=False):
    if verbose:
        print("Process start with parameters : subject=", subject, ", experience=", n_experience)
    n_experience = int(n_experience)
    runs = experiments[n_experience]['runs']
    raw, _ = get_raw(subject = subject, n_experience=n_experience, runs=runs)
    
    if drop_option:
        bad_channels = raw.info['bads']
        raw = drop_bad_channels(raw, bad_channels, verbose)
    
    raw = my_filter(raw, verbose)
    # Read epochs (events)
    epochs_train, labels = get_data(raw)
    
    cv = ShuffleSplit(10, test_size=0.2, random_state=0)
    # Assemble a classifier #2
    csp = CSP(6)
    lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    clf= Pipeline([("CSP", csp), ("LDA", lda)], verbose=False)
    
    # fit our pipeline to the experiment #1
    _, X_test, _, y_test = train_test_split(epochs_train, labels, random_state=0)
    if verbose == False:
        default_stdout = sys.stdout
        # Rediriger la sortie vers null
        sys.stdout = open('/dev/null', 'w')

    clf.fit(epochs_train, labels)
    predictions =clf.predict(X_test)
    score = accuracy_score(predictions, y_test)
    if verbose == False:
        # Restaurer la sortie par défaut
        sys.stdout = default_stdout
    else:
        title = "Learning Curves "
        plot_learning_curve(clf, title, epochs_train, labels,cv=cv, n_jobs=-1)
        plt.show()
        print(f"Training with Patient #{colors.blue}{subject}{colors.reset} [{colors.green}{experiments[n_experience]['description']}{colors.reset}] ... Done")

    #Save
    save_model(clf, get_path(subject, n_experience), verbose=verbose)
    
    return score
