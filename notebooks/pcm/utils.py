import os
import inspect
import pandas as pd
import numpy as np
from functools import reduce

from sklearn.preprocessing import StandardScaler

from typing import Union, TypeVar

PathLike = TypeVar("PathLike")
DataFrameLike = TypeVar("DataFrameLike")
ScalerClassLike = TypeVar("ScalerClassLike")

#from sklearn.base import BaseEstimator, TransformerMixin

class ZScalesExtractor:
    def __init__(self, zscales: Union[DataFrameLike, PathLike], calc_cross_term: bool=True,
                 scaler: Union[ScalerClassLike, None]=None):
        if isinstance(zscales, pd.DataFrame):
            self.__zscales_df = zscales
        elif os.path.isfile(zscales):
            self.__zscales_df = pd.read_csv(zscales)
        else:
            raise Exception("Invalid argument for zscales")
        self.zscale_dict = dict(zip(self.__zscales_df["1-letter-code"],
                                    self.__zscales_df[["z1", "z2", "z3", "z4", "z5"]].values.tolist()
                                   )
                               )
        if scaler is None:
            self.peptide_scaler = StandardScaler()
            self.target_scaler = StandardScaler()
        elif callable(scaler):
            if inspect.isclass(scaler):
                if hasattr(scaler, "fit") and hasattr(scaler, "transform"):
                    self.peptide_scaler = scaler()
                else:
                    raise Exception("scaler class must have fit and transform methods")
            else:
                raise Exception("scaler must be a class with fit and transform methods")
        else:
            raise Exception("scaler must be a class with fit and transform methods")
        
        self.calc_cross_term = calc_cross_term
        self.__scaler_fitted = False

    def fit(self, X, y):
        return self
    
    def transform(self, X, y = None):
        peptide_zscales = X.apply(lambda row: self.__get_seq_zscales(row["Sequence"]), axis=1, result_type='expand')
        target_zscales = X.apply(lambda row: self.__get_seq_zscales(row["Target_seq"]), axis=1, result_type='expand')
        
        if not self.__scaler_fitted:
            _ = self.peptide_scaler.fit(peptide_zscales)
            _ = self.target_scaler.fit(target_zscales)
            self.__scaler_fitted = True
        
        peptide_zscales = self.peptide_scaler.transform(peptide_zscales)
        peptide_zscales = pd.DataFrame(peptide_zscales, columns=["p_z"+str(i) for i in range(1,6)])
        
        target_zscales = self.target_scaler.transform(target_zscales)
        target_zscales = pd.DataFrame(target_zscales, columns=["t_z"+str(i) for i in range(1,6)])
        
        return self.__get_zscale_features(peptide_zscales, target_zscales)
    
    def __get_aa_zscales(self, aa, zscale_df=None):
        # row = zscale_df[zscale_df["1-letter-code"] == aa]
        # return row[["z1", "z2", "z3", "z4", "z5"]].values.tolist()[0]
        # very slow with above code. roughly 1000x faster with dict
        # because searching in dict is only O(1) operation
        return self.zscale_dict[aa]
    
    def __get_seq_zscales(self, seq):
        seq = list(seq)
        #seq_zscales = reduce(lambda l1, l2: [a+b for a, b in zip(l1, l2)], map(get_aa_zscales, s))
        zscales_values = np.array(list(map(self.__get_aa_zscales, seq)))
        return zscales_values.mean(axis=0)
    
    def __cross_seq_term_zscales(self, zscales1, zscales2):
        return np.multiply.outer(zscales1,zscales2).flatten().tolist()
    
    def __get_zscale_features(self, peptide_zscales, target_zscales):
    
        assert peptide_zscales.shape == target_zscales.shape

        zscale_features = pd.concat([peptide_zscales, target_zscales], axis=1)
        
        if self.calc_cross_term:
            tmp = zscale_features.apply(lambda row: self.__cross_seq_term_zscales(
                row[["p_z1", "p_z2", "p_z3", "p_z4", "p_z5"]].values,
                row[["t_z1", "t_z2", "t_z3", "t_z4", "t_z5"]].values
            ), axis=1, result_type='expand')
            cross_term_cols = ["x_z"+str(i) for i in range(1,26)]
            zscale_features[cross_term_cols] = tmp

        return zscale_features