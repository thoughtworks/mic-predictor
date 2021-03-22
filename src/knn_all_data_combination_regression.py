import biovec
import numpy as np
import pandas as pd
from itertools import chain, combinations
from collections import Counter

from utils import *

from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

from Bio.SeqUtils.ProtParam import ProteinAnalysis

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mape_scorer = make_scorer(mean_absolute_percentage_error, greater_is_better=False)

def get_physicochemical_properties(df):
    params = ['aromaticity', 'helix', 'turn', 'sheet', 'gravy', 'net_charge_at_pH7point4']

    prop = []
    for seq in df.Sequence:
        X = ProteinAnalysis(seq)
        aromaticity = X.aromaticity()
        sec_struc = X.secondary_structure_fraction()
        helix = sec_struc[0]
        turn = sec_struc[1]
        sheet = sec_struc[2]
        gravy = X.gravy() # hydrophobicity related
        net_charge_at_pH7point4 = X.charge_at_pH(7.4)

        prop.append([aromaticity, helix, turn, sheet, gravy, net_charge_at_pH7point4])
    return pd.DataFrame(prop, columns=params)

def get_score(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    
    return (mse, mae, mape)

def get_tuned_model(clf, param_grid, X_train, y_train, n_jobs):
    grid_search = GridSearchCV(estimator = clf, param_grid = param_grid, 
                              cv = 5, n_jobs = n_jobs, verbose = 0, scoring=mape_scorer)
    
    _ = grid_search.fit(X_train, y_train)
    
    return grid_search

def get_tuned_regressor_train_test_scores(X, y, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    max_n_neighbors = int(np.sqrt(X_train.shape[0]))
    param_grid = {
                    'n_neighbors': range(1, max_n_neighbors),
                    'weights': ['uniform', 'distance'],
                    'metric': ["euclidean", "manhattan", "chebyshev"]
                }
    #param_grid = {'C':[100]}

    knn = KNeighborsRegressor()
    grid_search = get_tuned_model(knn, param_grid, X_train, y_train, n_jobs=10)
    best_grid = grid_search.best_estimator_
    best_grid.fit(X_train, y_train)
    y_pred = best_grid.predict(X_test)
    return grid_search.best_params_, get_score(y_test, y_pred)
    
def main():  
    avp_ic50 = pd.read_csv("../data/raw/AVP-IC50Pred_train.csv")
    ha_avp = pd.read_csv("../data/raw/HA_AVP.csv")
    shannon_entropy = pd.read_csv("../data/pfeature/ha_avp_ic50_shannon_entropy.csv")
    shannon_entropy = shannon_entropy.drop(['ID', 'Sequence'], axis=1)
    residue_repeats = pd.read_csv("../data/pfeature/ha_avp_ic50_residue_repeat.csv")
    residue_repeats = residue_repeats.drop(['ID', 'Sequence'], axis=1).reset_index(drop=True)

    df = pd.concat([avp_ic50[['Sequence','MIC']], ha_avp], axis=0).drop_duplicates(['Sequence']).reset_index(drop=True)
    df = sequence_filtering(df)

    ############# AA freq #############
    aa_freq = reduce_by_kmer_frequency(df)

    ############# Dipep freq #############
    dipep_freq = reduce_by_kmer_frequency(df, kmer=2)

    ############# Tripep freq #############
    # tripep_freq = reduce_by_kmer_frequency(df, kmer=3)

    ############# Prot2Vec #############
    uniprot_embedding = biovec.models.load_protvec("../data/embeddings/uniprot__kmer_3_contextWindow_10_vector_100_reduction_None")

    avg_protvec = convert_sequences_to_avg_vectors(df['Sequence'], uniprot_embedding, kmer=3)
    avg_protvec = avg_protvec.reset_index(drop=True)

    ############# physicochemical properties #############
    physicochemical_prop = get_physicochemical_properties(df)
    # physicochemical_prop = pd.concat([physicochemical_prop, shannon_entropy, residue_repeats], axis=1)

    data_dict = {
        "aa_freq": aa_freq,
        "dipep_freq": dipep_freq,
        # "tripep_freq": tripep_freq,
        "avg_protvec": avg_protvec,
        # "physicochemical_prop": physicochemical_prop,
        "shannon_entropy": shannon_entropy,
        "residue_repeats": residue_repeats
    }

    data_dict.update({prop: pd.DataFrame(physicochemical_prop[prop]) for prop in physicochemical_prop.columns})

    base_data_list = data_dict.copy().keys()
    
    for i in range(2, len(base_data_list)+1):
        data_combination = list(combinations(base_data_list, i))
        for combo in data_combination:
            data_dict["+".join(combo)] = pd.concat([data_dict[data] for data in combo], axis=1)

    import os

    res_file = "../results/knn_regressor_all_data_combination_separate_props_mic.csv"
    if not os.path.exists(res_file):
        with open(res_file, "w") as f:
            header = ", ".join(["Data", "Regressor", "Best parameters", 
                            "MSE", "MAE","MAPE"])+"\n"
            f.write(header)
    else:
        raise FileExistsError("Result file already exists.")

    for data_name, data in data_dict.items():
        print(f"#################################### {data_name} ####################################")
        best_params, scores = get_tuned_regressor_train_test_scores(data, df['MIC'])
        data_to_write = data_name + '; ' + "KNN" + '; ' + '"'+str(best_params)+'"' + '; ' + '; '.join([str(s) for s in scores]) + '\n'
        
        with open(res_file, "a") as f:
            f.write(data_to_write)
    
if __name__ == "__main__":
    main()
    
