from tqdm import tqdm
import biovec
import numpy as np
import pandas as pd
from itertools import chain, product
from collections import Counter

AMINO_ACID_RESIDUES = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

def prot_vec_to_vecs(pv, x, k):
    return pv.to_vecs(x)

def split_n_grams(seq, n):
    """
    'AGAMQSASM' => [['AGA', 'MQS', 'ASM'], ['GAM','QSA'], ['AMQ', 'SAS']]
    In case of n = 3
    """
    grams = []
    for i in range(n):
        grams.append(zip(*[iter(seq[i:])] * n))

    str_ngrams = []
    for ngrams in grams:
        x = []
        for ngram in ngrams:
            x.append("".join(ngram))
        str_ngrams.append(x)
    return str_ngrams

def words_to_vec(pv, seq, n=5):
    ngram_patterns = split_n_grams(seq, n)

    vectors = []
    for ngrams in ngram_patterns:
        ngram_vecs = []
        for ngram in ngrams:
            try:
                ngram_vecs.append(pv[ngram])
            except:
                print(ngram)
                raise Exception("Model has never trained this n-gram: " + ngram)
        vectors.append(sum(ngram_vecs))
    return vectors

def dipeptide_encoding(seq, n):
    """
    Returns n-Gram Motif frequency
    https://www.biorxiv.org/content/10.1101/170407v1.full.pdf
    """
    aa_list = list(seq)
    return {''.join(aa_list): n for aa_list, n in Counter(zip(*[aa_list[i:] for i in range(n)])).items() if
            not aa_list[0][-1] == (',')}

def get_kmer_list(kmer):
    return ["".join(s) for s in product(AMINO_ACID_RESIDUES, repeat=kmer)]

def reduce_by_kmer_frequency(data, kmer=1):
    seq_vec = data.Sequence.apply(lambda x: dipeptide_encoding(x, kmer)).to_list()
    df = pd.DataFrame(seq_vec)
    df = df.fillna(0)
    df = df.reindex(columns=get_kmer_list(kmer), fill_value=0)
    return df.div(df.sum(axis=1), axis=0)

def convert_sequences_to_vectors(data, embedding, to_vec=prot_vec_to_vecs, kmer=5):
    output = pd.DataFrame()
    errors = list()
    for row in tqdm(data, desc="Creating vectors", unit="sequence"):
        try:
            output = output.append(pd.DataFrame(sum(to_vec(embedding, row, kmer))).T)
        except:
            output = output.append(pd.DataFrame(np.zeros((1, embedding.vector_size))))
            errors.append(row)
    return output, errors

OTHER_ALPHABETS = "UOXBZJ"

def convert_sequences_to_avg_vectors(data, embedding, kmer_wt=None, kmer=3):
    output = pd.DataFrame()
    if kmer_wt is None:
        for row in tqdm(data, desc="Creating vectors", unit="sequence"):
            vec = []
            ngrams = list(chain(*split_n_grams(row, kmer)))
            for ngram in ngrams:
                try:
                    vec.append(embedding[ngram])
                except:
                    vec.append(np.zeros(embedding.vector_size))
            output = output.append(pd.DataFrame(np.mean(vec, axis=0)).T)
    else:
        for i, row in tqdm(enumerate(data), desc="Creating vectors", unit="sequence"):
            vec = []
            ngrams = list(chain(*split_n_grams(row, kmer)))
            for ngram in ngrams:
                try:
                    vec.append(embedding[ngram] * kmer_wt[ngram][i])
                except:
                    vec.append(np.zeros(embedding.vector_size))
            output = output.append(pd.DataFrame(np.mean(vec, axis=0)).T)
    return output

def contains(other_alphabets, seq):
    for o in str(other_alphabets):
        if o in str(seq):
            return True
    return False

def sequence_filtering(data):
    sequences = data[data.apply(lambda r: not contains(OTHER_ALPHABETS, r['Sequence']), axis=1)]
    sequences = sequences[sequences.apply(lambda r: not str(r['Sequence']) == 'nan', axis=1)]
    sequences['Sequence'] = sequences['Sequence'].apply(lambda x: x.upper())
    sequences['Sequence'] = sequences['Sequence'].apply(lambda x: x.strip())
    return sequences