import sys
import matplotlib.pyplot as plt
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.metrics import accuracy_score
from mne.decoding import CSP 

from .experiments import experiments
from .utils_raw import *
from .commun import colorize
from .graph import plot_learning_curve



def train(subject:int, n_experience:int, drop_option, verbose=False):
    if verbose:
        print("Process start with parameters : subject=", subject, ", experience=", n_experience)
    n_experience = int(n_experience)
    raw, _ = get_raw(subject = subject, n_experience=n_experience, drop_option=drop_option)
    
    if drop_option:
        raw = drop_bad_channels(raw=raw, name=get_name_model(subject=subject, n_experience=n_experience),save=False , verbose=verbose)
    
    raw = my_filter(raw, verbose)
    
    X_train, _, y_train, _ = perso_splitter(raw)
    
    # Assemble a classifier #2
    csp = CSP(6)
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
        # plt.show()
        print(f"Training with Patient #{colors.blue}{subject}{colors.reset} [{colors.green}{experiments[n_experience]['description']}{colors.reset}] ... Done")

    #Save
    clf.drop_option = drop_option
    save_model(clf, get_path(subject, n_experience), verbose=verbose)
    return mean_cvs
