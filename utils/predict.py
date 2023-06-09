import sys
from sklearn.metrics import accuracy_score

from .utils_raw import get_raw, drop_bad_channels, my_filter, perso_splitter
from .commun import colors
from .utils_models import get_name_model

def predict(subject:int, n_experience:int, model=None, verbose = False):
    subject = int((subject))
    n_experience = int(n_experience)
    if model is None:
        return None
    drop_option = model.drop_option

    raw, _,_ = get_raw(subject = subject, n_experience=n_experience, drop_option=drop_option)
    if drop_option:
        raw = drop_bad_channels(raw=raw, name=get_name_model(subject=subject, n_experience=n_experience),save=False , verbose=False)

    # Apply band-pass filter
    raw = my_filter(raw)

    _, X_test, _, y_test = perso_splitter(raw)

    default_stdout = sys.stdout
    # Rediriger la sortie vers null
    sys.stdout = open('/dev/null', 'w')
    try:
        predictions = model.predict(X_test)
    except Exception:
        # Restaurer la sortie par défaut
        sys.stdout = default_stdout
        print(f"{colors.red} -> Changes in Parameters between training and prediction.{colors.reset}", end='')
        return None
    #accuracy score
    score_subject = accuracy_score(predictions, y_test)

    # Restaurer la sortie par défaut
    sys.stdout = default_stdout

    return score_subject