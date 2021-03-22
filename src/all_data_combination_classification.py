import biovec
import numpy as np
import pandas as pd
from itertools import chain
from collections import Counter

from utils import *

from sklearn.svm import SVC
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, matthews_corrcoef, precision_score, recall_score
from sklearn.metrics import classification_report, confusion_matrix, make_scorer, precision_recall_curve, auc
from sklearn.model_selection import train_test_split

from Bio.SeqUtils.ProtParam import ProteinAnalysis

mcc_scorer = make_scorer(matthews_corrcoef)

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

def get_precision_recall_scores(y_true, y_pred_probs):
    precision, recall, _ = precision_recall_curve(y_true, y_pred_probs)
    auprc = auc(recall, precision)
    return auprc

def get_score(y_true, y_pred, y_pred_probs):
    acc = accuracy_score(y_true, y_pred)
    auprc = get_precision_recall_scores(y_true, y_pred_probs)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auroc = roc_auc_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    return (acc, precision, recall, f1, auprc, auroc, tn, fp, fn, tp, mcc)

def get_tuned_classifier(clf, param_grid, X_train, y_train, n_jobs):
    grid_search = GridSearchCV(estimator = clf, param_grid = param_grid, 
                              cv = 5, n_jobs = n_jobs, verbose = 0, scoring=mcc_scorer)
    
    _ = grid_search.fit(X_train, y_train)
    
    return grid_search

def get_tuned_classifier_train_test_scores(X, y, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    param_grid = {'C':[0.01,0.1,1,100,1000],
                  'kernel': ['rbf','poly','sigmoid','linear'],
                  'degree': [1,2,3,4,5,6],
                  'gamma': ['scale', 'auto', 100, 10, 1, 0.1, 0.01, 0.001, 0.0001],
                  'coef0': [0.0,0.5,1,1.5,2,2.5,3],
                  'class_weight': ['balanced', None]
                 }
    #param_grid = {'C':[100]}

    clf = SVC(probability=True)
    grid_search = get_tuned_classifier(clf, param_grid, X_train, y_train, n_jobs=10)
    best_grid = grid_search.best_estimator_
    best_grid.fit(X_train, y_train)
    y_pred = best_grid.predict(X_test)
    y_pred_probs = best_grid.predict_proba(X_test)
    y_pred_probs = y_pred_probs[:,1]
    return grid_search.best_params_, get_score(y_test, y_pred, y_pred_probs)
    
def main():  
    avp_ic50 = pd.read_csv("../data/raw/AVP-IC50Pred_train.csv")
    ha_avp = pd.read_csv("../data/raw/HA_AVP.csv")
    shannon_entropy = pd.read_csv("../data/pfeature/ha_avp_ic50_shannon_entropy.csv")
    shannon_entropy = shannon_entropy.drop(['ID', 'Sequence'], axis=1)
    residue_repeats = pd.read_csv("../data/pfeature/ha_avp_ic50_residue_repeat.csv")
    residue_repeats = residue_repeats.drop(['ID', 'Sequence'], axis=1).reset_index(drop=True)

    df = pd.concat([avp_ic50[['Sequence','MIC']], ha_avp], axis=0).drop_duplicates(['Sequence']).reset_index(drop=True)
    df = sequence_filtering(df)

    df['lessthan5'] = df['MIC'].apply(lambda mic: 1 if mic <= 5 else 0)

    ############# AA freq #############
    aa_freq = reduce_by_kmer_frequency(df)

    ############# Dipep freq #############
    dipep_freq = reduce_by_kmer_frequency(df, kmer=2)

    ############# Prot2Vec #############
    uniprot_embedding = biovec.models.load_protvec("../data/embeddings/uniprot__kmer_3_contextWindow_10_vector_100_reduction_None")

    avg_vectors = convert_sequences_to_avg_vectors(df['Sequence'], uniprot_embedding)
    avg_vectors = avg_vectors.reset_index(drop=True)

    ############# Prot2Vec + AA freq #############
    protvec_aa_freq = pd.concat([avg_vectors, aa_freq], axis=1)

    ############# Prot2Vec + dipep freq #############
    protvec_dipep_freq = pd.concat([avg_vectors, dipep_freq], axis=1)

    ############# Prot2Vec + AA freq + dipep freq #############
    protvec_aa_dipep = pd.concat([avg_vectors, aa_freq, dipep_freq], axis=1)

    ############# physicochemical properties #############
    physicochem_prop = get_physicochemical_properties(df)
    physicochem_prop = pd.concat([physicochem_prop, shannon_entropy, residue_repeats], axis=1)

    ############# AA freq + physicochemical properties #############
    aa_freq_physicochem_prop = pd.concat([aa_freq, physicochem_prop], axis=1)

    ############# Dipep freq + physicochemical properties #############
    dipep_freq_physicochem_prop = pd.concat([dipep_freq, physicochem_prop], axis=1)

    ############# AA freq + Dipep freq + physicochemical properties #############
    aa_dipep_freq_physicochem_prop = pd.concat([aa_freq, dipep_freq, physicochem_prop], axis=1)

    ############# Prot2Vec + physicochemical properties #############
    protvec_physicochem_prop = pd.concat([avg_vectors, physicochem_prop], axis=1)

    ############# Prot2Vec + AA freq + physicochemical properties #############
    protvec_aa_freq_physicochem_prop = pd.concat([avg_vectors, aa_freq, physicochem_prop], axis=1)

    ############# Prot2Vec + AA freq + dipep freq + physicochemical properties #############
    protvec_aa_dipep_freq_physicochem_prop = pd.concat([avg_vectors, aa_freq, dipep_freq, physicochem_prop], axis=1)

    data_dict = {
        "aa_freq": avg_vectors,
        "dipep_freq": dipep_freq,
        "avg_protvec": avg_vectors,
        "protvec+aa_freq": protvec_aa_freq,
        "protvec+dipep_freq": protvec_dipep_freq,
        "protvec+aa_freq+dipep_freq": protvec_aa_dipep,
        "physicochemical_prop": physicochem_prop,
        "aa_freq+physicochemical_prop": aa_freq_physicochem_prop,
        "dipep_freq+physicochemical_prop": dipep_freq_physicochem_prop,
        "aa_freq+dipep_freq+physicochemical_prop": aa_dipep_freq_physicochem_prop,
        "protvec+physicochemical_prop": protvec_physicochem_prop,
        "protvec+aa_freq+physicochemical_prop": protvec_aa_freq_physicochem_prop,
        "protvec+aa_freq+dipep_freq+physicochemical_prop": protvec_aa_dipep_freq_physicochem_prop
    }

    import os

    res_file = "../results/all_combination_classifier.csv"
    if not os.path.exists(res_file):
        with open(res_file, "w") as f:
            header = ", ".join(["Data", "Classifier", "Best parameters", 
                            "Accuracy", "Precision", "Recall", "F1-score", "AUPRC", "AUROC", 
                            "TN", "FP", "FN", "TP", "MCC"])+"\n"
            f.write(header)
    else:
        raise FileExistsError("Result file already exists.")

    for data_name, data in data_dict.items():
        print(f"#################################### {data_name} ####################################")
        best_params, scores = get_tuned_classifier_train_test_scores(data, df['lessthan5'])
        data_to_write = data_name + ', ' + "SVM" + ', ' + '"'+str(best_params)+'"' + ', ' + ', '.join([str(s) for s in scores]) + '\n'
        
        with open(res_file, "a") as f:
            f.write(data_to_write)
    
if __name__ == "__main__":
    main()
    