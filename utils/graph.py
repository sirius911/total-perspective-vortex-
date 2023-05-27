import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=5,
                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):
    
    # Rediriger la sortie vers null
    default_stdout = sys.stdout
    sys.stdout = open('/dev/null', 'w')

    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=np.linspace(0.1, 1.0, 10), scoring='accuracy', verbose=False)
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)

    # Restaurer la sortie par défaut
    sys.stdout = default_stdout
    
    plt.plot(train_sizes, train_scores_mean, label='Training score')
    plt.plot(train_sizes, test_scores_mean, label='Cross-validation score')
    plt.xlabel('Number of epochs')
    plt.ylabel('Accuracy score')
    plt.legend()
    plt.title(title)
    plt.show()