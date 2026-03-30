import numpy as np
import itertools
import pandas as pd

def get_trials(dataset_name):
    """
    Get all gain-loss combinations for a given dataset.
    """
    if dataset_name == "equalIndifference":
        gains = np.arange(10, 41, 2)
        losses = np.arange(5, 21, 1)
        combinations = list(itertools.product(gains, losses))
    elif dataset_name == "equalRange":
        gains = np.arange(5, 21, 1)
        losses = np.arange(5, 21, 1)
        combinations = list(itertools.product(gains, losses))
    elif dataset_name == "sheng2020":
        gains = np.arange(1, 11, 1)
        losses = np.arange(1, 11, 1)
        combinations = list(itertools.product(gains, losses))
    else:
        raise ValueError(f"Dataset {dataset_name} not recognized.")
    
    df = pd.DataFrame(combinations, columns=['gain', 'loss'])
    return df

def logistic(x):
    """
    Logistic function.
    """
    return 1 / (1 + np.exp(-x))

def probit(x):
    """
    Probit function.
    """
    from scipy.stats import norm
    return norm.cdf(x)