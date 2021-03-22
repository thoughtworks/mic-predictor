from tqdm import tqdm
import biovec
import numpy as np
import pandas as pd
from s

from src.utils import convert_sequences_to_vectors, words_to_vec, reduce_by_alphabet_frequency

#from src.utils import 

avp_ic50 = pd.read_csv("../data/raw/AVP-IC50Pred_train.csv")
ha_avp = pd.read_csv("../data/raw/HA_AVP.csv")

df = pd.concat([avp_ic50[['Sequence', 'MIC']], ha_avp], axis=0).drop_duplicates(['Sequence']).reset_index(drop=True)

############# Amino acid frequency #############
aa_freq = reduce_by_alphabet_frequency(df)


############# Prot2Vec #############
uniprot_embedding = biovec.models.load_protvec("data/embeddings/uniprot__kmer_3_contextWindow_10_vector_100_reduction_None")

vectors, _ = convert_sequences_to_vectors(df['Sequence'], uniprot_embedding, words_to_vec, kmer=3)
vectors = vectors.reset_index(drop=True)