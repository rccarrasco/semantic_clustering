# -*- coding: utf-8 -*-
"""
Created on Thu Jul  3 16:47:15 2025

@author: carrraf
"""
import pandas as pd
import numpy as np
import gzip

from scipy.sparse import lil_matrix, csr_matrix
from scipy.spatial.distance import cosine
from sklearn.neighbors import KDTree
from tqdm import tqdm
from sknetwork.clustering import Leiden
from sklearn.feature_extraction.text import CountVectorizer

n_docs = 40000
input_path = 'input/embeddings.npz'
data = np.load(input_path)['data_array'][:n_docs, :]
tree = KDTree(data, leaf_size=10)

# adjacency marix
shape = n_docs, n_docs
A = lil_matrix(shape, dtype=float)
for m in tqdm(range(n_docs), total=n_docs):
    X = data[m].reshape(1, -1)
    d, cols = tree.query(X, k=100)
    for n in cols[0]:
        A[m, n] = 1 - cosine(data[m], data[n])


# clusters the proposals
model = Leiden(resolution=10)
cluster_label = model.fit_predict(A.tocsr())
num_clusters = len(set(cluster_label))
print('Created', num_clusters, 'clusters')

# characterize clusters by lexical content
input_filename = 'input/titles.txt.gz'
with gzip.open('input/titles.txt.gz') as f:
    docs = [line.decode() for line in f.readlines()][:n_docs]


token_pattern = r'(?:\w+-)?[a-zA-Z]+(?:-\w+)*'
vectorizer = CountVectorizer(
    token_pattern=token_pattern,
    stop_words='english',
    max_df=0.1,
    min_df=5,
    max_features=20000,
    ngram_range=(1, 3))

term_frequency = vectorizer.fit_transform(docs)

num_docs, num_terms = term_frequency.shape
terms = vectorizer.get_feature_names_out()
print(num_terms, 'terms counted')

# compute cluster frequency using projection matrix
data = np.ones(num_docs)[:n_docs]
cluster_matrix = csr_matrix((data, (cluster_label, range(num_docs))))
term_frequency_per_cluster = cluster_matrix * term_frequency
av_tf = np.asarray(term_frequency_per_cluster.mean(axis=0)).reshape(-1)

# compute chi2 score of every term
columns = ['CLUSTER', 'TERM', 'SCORE']
res = pd.DataFrame(columns=columns)
for m in range(num_clusters):
    scores = list()
    for n, term in enumerate(terms):
        observed = term_frequency_per_cluster[m, n]
        expected = av_tf[n]
        score = (observed - expected) * (observed - expected) / expected
        scores.append((m, term, score))
    cluster_res = pd.DataFrame(scores, columns=columns)
    cluster_res.sort_values('SCORE', ascending=False, inplace=True)
    res = pd.concat([res, cluster_res.head(100)])

output_filename = 'output/cluster_terms.xlsx'
with pd.ExcelWriter(output_filename) as writer:
    for m in range(num_clusters):
        sel = res[res.CLUSTER == m].set_index('TERM')
        sel['SCORE'].to_excel(writer, sheet_name=f'cluster {m}')
print('Saved to', output_filename)
