import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import pearsonr
from sklearn.feature_selection import RFE
from sklearn.model_selection import cross_val_predict, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def get_scores(y_true, y_pred):
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    pcc = pearsonr(y_true, y_pred)[0]
    mse = mean_squared_error(y_true, y_pred)
    return {'mape': mape, 'r2': r2, 'pcc': pcc, 'mse': mse}

def get_adj_r2(r2, n, p):
    num = (1-r2)*(n-1)
    denom = (n-p-1)
    return 1 - (num/denom)

def get_loo_scores(data, target, model, features=None):
    X = data
    y = target
    if features is not None:
        X = data[features]
    y_pred = cross_val_predict(model, X, y, cv=X.shape[0], n_jobs=10)
    return get_scores(y, y_pred)

def adj_r2_feature_selection(data, target, model):
    rfe = RFE(model, n_features_to_select=5, step=1)
    rfe.fit(data, target)
    rfe_ranking_ = rfe.ranking_
    rank_sorted_features = sorted(data.columns, key=dict(zip(data.columns, rfe_ranking_)).get)
    best_adj_r2 = 0
    feature_lst = []
    best_scores = {}
    for i in tqdm(range(0, len(rank_sorted_features))):
        features = best_scores.get('features', []) + [rank_sorted_features[i]]
        loo_score = get_loo_scores(data, target, model, features)
        scores = {'features': features}
        scores.update(loo_score)
        adj_r2 = get_adj_r2(scores['r2'], data.shape[0], len(scores['features']))
        
        if adj_r2 > best_adj_r2:
            best_adj_r2 = adj_r2
            best_scores = scores
            best_scores['adj_r2'] = adj_r2
            
    return best_scores

def get_grid_search_best_params(data, target, model, param_grid, scoring, n_jobs=1, cv=5, verbose=2):
    grid = GridSearchCV(estimator = model, param_grid = param_grid, 
                        cv = cv, verbose=verbose, scoring = scoring, n_jobs = n_jobs)

    _ = grid.fit(data, target)
    return grid.best_params_