import os
import glob
from tqdm import tqdm
import pickle
import biovec
import numpy as np
import pandas as pd
from multiprocessing import Pool
from orderedset import OrderedSet
from itertools import chain, combinations
from functools import reduce
from collections import Counter

from scipy.stats import pearsonr

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
    pearson_corrcoef = pearsonr(y_true, y_pred)
    
    return {"MSE":mse, "MAE":mae,"MAPE":mape, "PCC":pearson_corrcoef}

def get_tuned_model(clf, param_grid, X_train, y_train, n_jobs):
    grid_search = GridSearchCV(estimator = clf, param_grid = param_grid, 
                              cv = 5, n_jobs = n_jobs, verbose = 0, scoring=mape_scorer)
    
    _ = grid_search.fit(X_train, y_train)
    
    return grid_search

def get_net_contact_energy(df):
    contact_energy_matrix = pd.read_csv("../data/raw/contact_energy.csv", index_col=0)
    code3_to_code1 = pd.read_excel("../data/raw/amino_acid_codes.xlsx")
    code3_to_code1_dict = dict(code3_to_code1[['Three-letter symbol', 'One-letter symbol']].values)
    contact_energy_matrix.columns = [code3_to_code1_dict[code] for code in contact_energy_matrix.columns]
    contact_energy_matrix.index = [code3_to_code1_dict[code] for code in contact_energy_matrix.index]
    def seq_to_contact_energy(seq):
        return [contact_energy_matrix[r][c] for r,c in list(zip(seq[:-1],seq[1:]))]
    df['contact_energy'] = df['Sequence'].apply(seq_to_contact_energy)
    df['net_contact_energy'] = df['contact_energy'].apply(lambda s: np.sum(s))
    return df

def combine_df(df1, df2):
    a = df2['aaindex'][0]
    df2 = df2.drop(['Sequence', 'aaindex'], axis=1)
    df2.columns = [col+'_'+a for col in df2.columns]
    return pd.concat([df1, df2], axis=1)

def get_autocorr_prop(path):
    gg = reduce(combine_df, map(pd.read_csv, glob.glob(path)))
    gg.rename(columns = {col: col+'_'+gg['aaindex'][0] for col in ['ACR1_MB','ACR1_MO','ACR1_GE']}, inplace=True)
    return gg.drop(['Sequence', 'aaindex'], axis=1)

def get_tuned_regressor_train_test_scores(X, y, data_name, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    max_n_neighbors = int(np.sqrt(X_train.shape[0]))
    param_grid = {
                    'n_neighbors': range(1, max_n_neighbors),
                    'weights': ['uniform', 'distance'],
                    'metric': ["euclidean", "manhattan", "chebyshev"]
                }
    #param_grid = {'C':[100]}

    knn = KNeighborsRegressor()
    grid_search = get_tuned_model(knn, param_grid, X_train, y_train, n_jobs=1)
    best_grid = grid_search.best_estimator_
    best_grid.fit(X_train, y_train)
    y_pred = best_grid.predict(X_test)
    result = {'data':data_name, 'Best_params':grid_search.best_params_}
    result.update(get_score(y_test, y_pred))
    return result

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def get_data_combination(combination, data_dict):
    name = "+".join(combination)
    data = pd.concat([data_dict[data] for data in combination], axis=1)
    return name, data

def run_regression_with_combination(combination, target, data_dict, random_state=42):
    name, data = get_data_combination(combination, data_dict)
    return get_tuned_regressor_train_test_scores(data, target, name, random_state=random_state)

def main():  
    avp_ic50 = pd.read_csv("../data/raw/AVP-IC50Pred_train.csv")
    ha_avp = pd.read_csv("../data/raw/HA_AVP.csv")
    paac = pd.read_csv("../data/pfeature/paac.csv")
    paac = paac.drop(['ID'], axis=1)
    cetd = pd.read_csv("../data/pfeature/CeTD.csv")
    cetd = cetd.drop(['ID'], axis=1)
    shannon_entropy = pd.read_csv("../data/pfeature/ha_avp_ic50_shannon_entropy.csv")
    shannon_entropy = shannon_entropy.drop(['ID', 'Sequence'], axis=1)
    residue_repeats = pd.read_csv("../data/pfeature/ha_avp_ic50_residue_repeat.csv")
    residue_repeats = residue_repeats.drop(['ID', 'Sequence'], axis=1).reset_index(drop=True)

    df = pd.concat([avp_ic50[['Sequence','MIC']], ha_avp], axis=0).drop_duplicates(['Sequence']).reset_index(drop=True)
    df = sequence_filtering(df)

    df = get_net_contact_energy(df)

    ############# AA freq #############
    aa_freq = reduce_by_kmer_frequency(df)

    ############# Dipep freq #############
    dipep_freq = reduce_by_kmer_frequency(df, kmer=2)

    ############# Prot2Vec #############
    uniprot_embedding = biovec.models.load_protvec("../data/embeddings/uniprot__kmer_3_contextWindow_10_vector_100_reduction_None")

    avg_protvec = convert_sequences_to_avg_vectors(df['Sequence'], uniprot_embedding, kmer=3)
    avg_protvec = avg_protvec.reset_index(drop=True)

    ############# physicochemical properties #############
    physicochemical_prop = get_physicochemical_properties(df)
    # physicochemical_prop = pd.concat([physicochemical_prop, shannon_entropy, residue_repeats], axis=1)

    ############# autocorrelation properties #############
    #autocorr_prop = get_autocorr_prop("../data/pfeature/Autocorrelation_prop/*")

    data_dict = {
        "aa_freq": aa_freq,
        "dipep_freq": dipep_freq,
        "avg_protvec": avg_protvec,
        "shannon_entropy": shannon_entropy,
        "residue_repeats": residue_repeats,
        "paac": paac,
        "cetd": cetd,
        "net_contact_energy": pd.DataFrame(df['net_contact_energy'])
    }

    data_dict.update({prop: pd.DataFrame(physicochemical_prop[prop]) for prop in physicochemical_prop.columns})

    #data_dict.update({prop: pd.DataFrame(autocorr_prop[prop]) for prop in autocorr_prop.columns})

    base_data_list = data_dict.copy().keys()
    
    all_data_combinations = []
    for i in range(1, len(base_data_list)+1):
        all_data_combinations.extend(list(combinations(base_data_list, i)))


    res_file = "../results/test_test_knn_regressor_with_pfeature_all_data_combination_separate_props_mic.csv"
    if not os.path.exists(res_file):
        pd.DataFrame(columns=["Data", "Best_params", "MSE", "MAE", "MAPE", "PCC"]).to_csv(res_file, 
                                                                                        index=False, sep=';')
    else:
        raise FileExistsError("Result file already exists.")
    
    completed_res = pd.read_csv("../results/test_knn_regressor_with_pfeature_all_data_combination_separate_props_mic.csv", sep=';')
    done_combinations = OrderedSet(completed_res.Data)
    
    pool = Pool(processes=10)
    
    results = []
    chunk_size = 10
    for reduction_chunks in tqdm(chunks(all_data_combinations, chunk_size), desc="Chuncks completed", unit="chunk", total=int(len(all_data_combinations)/chunk_size)):
        chunk_set = OrderedSet("+".join(combo) for combo in reduction_chunks)
        remaining_combination = chunk_set.difference(done_combinations)
        if not bool(remaining_combination):
            continue
        else:
            remaining_combination = [combination.split('+') for combination in remaining_combination]
            futures = [pool.apply_async(run_regression_with_combination, args=(combination, df['MIC'], data_dict)) for combination in remaining_combination]
            res = [p.get() for p in futures]
            pd.DataFrame(res).to_csv(res_file, mode="a", header=False, index=False, sep=';')
    
if __name__ == "__main__":
    main()
