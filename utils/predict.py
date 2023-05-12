import sys
import numpy as np
from sklearn.model_selection import train_test_split, ShuffleSplit, cross_val_score
from sklearn.metrics import accuracy_score

from .experiments import experiments
from .utils_raw import *

def predict(subject:int, n_experience:int, drop_option, verbose = False):
    print("Process start with parameters : subject=", subject, ", experience=", n_experience)
    
    subject = int((subject))
    n_experience = int(n_experience)
    runs = experiments[n_experience]['runs']
    raw, _ = get_raw(subject = subject, n_experience=n_experience, runs=runs)
    if drop_option:
        bad_channels = raw.info['bads']
        raw = drop_bad_channels(raw, bad_channels, verbose)

    # Apply band-pass filter
    raw = my_filter(raw,verbose=verbose)

    cv = ShuffleSplit(10, test_size=0.2, random_state=0)

    # Read epochs
    epochs_train, labels = get_data(raw)
    _, X_test, _, y_test = train_test_split(epochs_train, labels, random_state=0)

    model = load_model(get_path(subject, n_experience))
    if verbose == False:
        default_stdout = sys.stdout
        # Rediriger la sortie vers null
        sys.stdout = open('/dev/null', 'w')
    predictions = model.predict(X_test)
    scores_ldashrinkage = cross_val_score(model, epochs_train, labels, cv=cv, n_jobs=-1, verbose=0)
    mean_scores_ldashrinkage = np.mean(scores_ldashrinkage)

    if verbose == False:
        # Restaurer la sortie par défaut
        sys.stdout = default_stdout
    else:
        print(f'event nb: [prediction] [truth] equal?')
        for i, prediction in enumerate(predictions):
            print(f'event {i:02d}: [{prediction}] [{y_test[i]}] {valid(prediction == y_test[i])}')
        score_subject = accuracy_score(predictions, y_test)
        print(f'Mean accuracy for all experiments:{score_subject}')
        print(f"Mean cross val score {mean_scores_ldashrinkage}")
    return mean_scores_ldashrinkage