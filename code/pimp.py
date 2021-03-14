import numpy as np

from sklearn.metrics import get_scorer
from scipy.stats import norm

from datetime import datetime as time


def _permutation_score_i(model, X, y, scorer, n_repeats, i_col):
    """
    Calculate permutation score when 'i_col' is permutated.
    """
    X_per = X.copy()
    scores = np.zeros(n_repeats)
    for i in range(n_repeats):
        np.random.shuffle(X_per)
        
        X[:, i_col] = X_per[:, i_col]
        if scorer:
            score = scorer(model, X, y)
        else:
            score = model.score(X, y)
        
        scores[i] = score

    return np.array(scores)

def permutation_importance(model, X, y, scoring=None, n_repeats=5):
    """
    Permutation importance for feature evaluation.
    
    Parameters
    ----------
    model : object
        A fitted model
    X : array , shape (n_samples, n_features)
        Data on which permutation importance will be computed.
    y : array, shape (n_samples)
        Targets.
    scoring : string, default=None
        Scorer to use.
        If None, the model's default scorer is used.
        Scorer options may be found here:
        https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
    n_repeats : int, default=5
        Number of times to permute a feature.
   
    Returns
    -------
    result : dictionary
        Dictionary, where the attributes are the following.
        importance : array, shape (n_features, n_repeats)
            Feature importance with n_repeats per feature
        p_vals : array, shape (n_features, )
            Feature importance by p-value of each feature
        mean : array, shape (n_features, )
            Feature importance's mean over `n_repeats`.
        std : array, shape (n_features, )
            Feature importance's standard deviation over `n_repeats`.
    
    Reference
    ---------
    Classic permutation importance:
    L. Breiman, "Random Forests", Machine Learning, 45(1), 5-32,
    2001. https://doi.org/10.1023/A:1010933404324

    P-value permutation importance:
    Altmann A, Toloşi L, Sander O, Lengauer T. Permutation importance: a corrected feature
    importance measure. Bioinformatics 2010, 26:1340–1347. 
    https://academic.oup.com/bioinformatics/article/26/10/1340/193348

    """
    if scoring:
        scorer = get_scorer(scoring)
        base_scores = scorer(model, X, y)
    
    else:
        scorer = None
        base_scores = model.score(X, y)
    

    scores = [_permutation_score_i(model, X, y, scorer, n_repeats, i_col) for i_col in range(X.shape[1])]
    importances = base_scores - scores

    means = np.mean(importances, axis=1)
    stds = np.std(importances, axis=1)

    # null_prob - probability distribution over the importances
    null_mean = np.mean(importances.reshape(-1))
    null_std = np.std(importances.reshape(-1))
    null_prob = norm(loc=null_mean, scale=null_std)

    p_vals = np.array([ 1 - null_prob.cdf(_mean) for _mean in means ])
    
    return {
            'importances': importances,
            'p_vals' : p_vals,
            'mean': means,
            'std': stds
           }
