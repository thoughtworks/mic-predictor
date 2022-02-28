import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer

from zscale_extractor import ZScalesExtractor
from utils import mean_absolute_percentage_error, adj_r2_feature_selection, \
                     get_loo_scores, get_grid_search_best_params

mic = pd.read_csv("../data/raw/MIC_pIC50_values.csv")
seq = pd.read_csv("../data/raw/peptide_target_seq.csv")
seq = seq.merge(mic, on='Sequence')
zscale = ZScalesExtractor("../data/raw/z_scales_5.csv", calc_cross_term=False)
zscales_features = zscale.transform(seq)

# mape_scorer = make_scorer(mean_absolute_percentage_error, greater_is_better = False)

# rf = RandomForestRegressor()
# param_grid = {'bootstrap': [True, False],
#               'max_depth': [2, 3, 5, 10, 15, 20, None],
#               'min_samples_leaf': [1, 2, 4],
#               'min_samples_split': [2, 5, 10],
#               'n_estimators': [100, 200, 250, 300, 500, 1000, 1500]}
# best_params_ = get_grid_search_best_params(zscales_features, seq['pIC50'], rf, param_grid,
#                                             mape_scorer, n_jobs=10, cv=5, verbose=2)

params_ = {'bootstrap': False,
 'max_depth': 2,
 'min_samples_leaf': 1,
 'min_samples_split': 10,
 'n_estimators': 200}

rf = RandomForestRegressor(**params_)

best_scores = adj_r2_feature_selection(zscales_features, seq['pIC50'], rf)        
print(best_scores)

features = best_scores['features']
loo_score = get_loo_scores(zscales_features, seq['pIC50'], rf, features)
print(loo_score)