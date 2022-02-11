import pandas as pd
import numpy as np
from scipy.stats import pearsonr

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import make_scorer, r2_score, mean_squared_error
from sklearn.feature_selection import RFE

from notebooks.pcm.utils import ZScalesExtractor

mic = pd.read_csv("../data/raw/MIC_pIC50_values.csv")
seq = pd.read_csv("../data/raw/peptide_target_seq.csv")
seq = seq.merge(mic, on='Sequence')
zscale = ZScalesExtractor("../data/raw/z_scales_5.csv")
zscales_features = zscale.transform(seq)

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mape_scorer = make_scorer(mean_absolute_percentage_error, greater_is_better = False)

params_ = {'bootstrap': False,
 'max_depth': 2,
 'min_samples_leaf': 1,
 'min_samples_split': 10,
 'n_estimators': 200}

rf = RandomForestRegressor(**params_)
rfe = RFE(rf, n_features_to_select=5, step=1)
rfe.fit(zscales_features, seq['pIC50'])

rfe_ranking_ = rfe.ranking_
# [31, 30, 29, 28,  1, 12,  1, 11,  1, 10,  9,  8, 14, 16, 18, 20, 22,
#                 24, 26, 27, 25, 23, 21, 19, 17, 15, 13,  7,  6,  5,  1,  2,  3,  1,
#                 4]

def get_scores(y_true, y_pred):
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    pcc = pearsonr(y_true, y_pred)[0]
    mse = mean_squared_error(y_true, y_pred)
    return {'mape': mape, 'r2': r2, 'pcc': pcc, 'mse': mse}

loo = LeaveOneOut()
from tqdm import tqdm

rank_sorted_features = sorted(zscales_features.columns, key=dict(zip(zscales_features.columns, rfe_ranking_)).get)

def get_adj_r2(r2, n, p):
    num = (1-r2)*(n-1)
    denom = (n-p-1)
    return 1 - (num/denom)

best_adj_r2 = 0
feature_lst = []
best_scores = {}
for i in tqdm(range(0, len(rank_sorted_features))):
    rf = RandomForestRegressor(**params_)
    features = best_scores.get('features', []) + [rank_sorted_features[i]]
    X = zscales_features[features]
    y = seq['pIC50']
    y_pred_lst = []
    y_test_lst = []
    for train_index, test_index in loo.split(zscales_features):
        X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        rf = RandomForestRegressor(**params_)
        _ = rf.fit(X_train, y_train)
        y_pred_lst.append(rf.predict(X_test)[0])
        y_test_lst.append(y_test.values[0])

    scores = {'features': features}
    scores.update(get_scores(y_test_lst, y_pred_lst))
    adj_r2 = get_adj_r2(scores['r2'], zscales_features.shape[0], len(scores['features']))
    
    if adj_r2 > best_adj_r2:
        best_adj_r2 = adj_r2
        best_scores = scores
        best_scores['adj_r2'] = adj_r2

print(best_scores)

rf = RandomForestRegressor(**params_)
features = ['t_z2', 'x_z21']
X = zscales_features[features]
y = seq['pIC50']
y_pred_lst = []
y_test_lst = []
for train_index, test_index in loo.split(zscales_features):
    X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    rf = RandomForestRegressor(**params_)
    _ = rf.fit(X_train, y_train)
    y_pred_lst.append(rf.predict(X_test)[0])
    y_test_lst.append(y_test.values[0])

print(get_scores(y_test_lst, y_pred_lst))