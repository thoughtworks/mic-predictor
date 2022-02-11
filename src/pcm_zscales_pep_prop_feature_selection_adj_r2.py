import pandas as pd
import numpy as np
from scipy.stats import pearsonr

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, LeaveOneOut
from sklearn.metrics import make_scorer, r2_score, mean_squared_error
from sklearn.feature_selection import RFE

from notebooks.pcm.utils import ZScalesExtractor

mic = pd.read_csv("../data/raw/MIC_pIC50_values.csv")
seq = pd.read_csv("../data/raw/peptide_target_seq.csv")
prop = pd.read_csv("../data/raw/prop.csv")
seq = seq.merge(prop, on='Sequence')
seq = seq.merge(mic, on='Sequence')
zscale = ZScalesExtractor("../data/raw/z_scales_5.csv")
zscales_features = zscale.transform(seq)

data = pd.concat([zscales_features, seq[['molecular_weight', 'aromaticity',
       'instability_index', 'isoelectric_point', 'helix', 'turn', 'sheet',
       'with_reduced_cysteines', 'with_disulfid_bridges', 'gravy',
       'net_charge_at_pH7point4']]], axis=1)

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mape_scorer = make_scorer(mean_absolute_percentage_error, greater_is_better = False)

# param_grid = {'bootstrap': [True, False],
#               'max_depth': [2, 3, 5, 10, 15, 20, None],
#               'min_samples_leaf': [1, 2, 4],
#               'min_samples_split': [2, 5, 10],
#               'n_estimators': [100, 200, 250, 300, 500, 1000, 1500]}

# rf = RandomForestRegressor()
# grid = GridSearchCV(estimator = rf, param_grid = param_grid, 
#                     cv = 5, verbose=2, scoring = mape_scorer, n_jobs = 10)

# _ = grid.fit(data, seq['pIC50'])

# best_params_ = grid.best_params_

params_ = {'bootstrap': False,
 'max_depth': 2,
 'min_samples_leaf': 4,
 'min_samples_split': 2,
 'n_estimators': 1500}

rf = RandomForestRegressor(**params_)
rfe = RFE(rf, n_features_to_select=5, step=1)
rfe.fit(data, seq['pIC50'])
rfe_ranking_ = rfe.ranking_

    # [42, 41, 40, 39, 38, 37,  1, 12,  1, 11, 10,  9,  8,  7, 14, 16, 18,
    #             20, 22, 24, 26, 28, 30, 32, 34, 36, 35, 33, 31, 29,  1,  6,  5,  4,
    #             3,  2,  1,  1, 13, 15, 17, 19, 21, 23, 25, 27]

def get_scores(y_true, y_pred):
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    pcc = pearsonr(y_true, y_pred)[0]
    mse = mean_squared_error(y_true, y_pred)
    return {'mape': mape, 'r2': r2, 'pcc': pcc, 'mse': mse}

loo = LeaveOneOut()
from tqdm import tqdm

rank_sorted_features = sorted(data.columns, key=dict(zip(data.columns, rfe_ranking_)).get)

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
    X = data[features]
    y = seq['pIC50']
    y_pred = cross_val_predict(rf, X, y, cv=X.shape[0], n_jobs=10)
    scores = {'features': features}
    scores.update(get_scores(y, y_pred))
    adj_r2 = get_adj_r2(scores['r2'], data.shape[0], len(scores['features']))
    
    if adj_r2 > best_adj_r2:
        best_adj_r2 = adj_r2
        best_scores = scores
        best_scores['adj_r2'] = adj_r2
        
print(best_scores)

rf = RandomForestRegressor(**params_)
features = ['t_z2', 'x_z21', 'aromaticity']#['t_z2', 'x_z1', 'x_z12', 'x_z17', 'x_z19']
X = data[features]
y = seq['pIC50']
y_pred_lst = []
y_test_lst = []
for train_index, test_index in tqdm(loo.split(data)):
    X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    rf = RandomForestRegressor(**params_)
    _ = rf.fit(X_train, y_train)
    y_pred_lst.append(rf.predict(X_test)[0])
    y_test_lst.append(y_test.values[0])

print(get_scores(y_test_lst, y_pred_lst))