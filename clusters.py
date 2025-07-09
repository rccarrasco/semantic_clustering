# -*- coding: utf-8 -*-
"""
Created on Thu Jul  3 16:47:15 2025

@author: carrraf
"""
import json
import pandas as pd
import numpy as np
# import multiprocessing as mp

from scipy.sparse import lil_matrix, csr_matrix
from scipy.spatial.distance import cosine
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from sknetwork.clustering import Leiden
from sklearn.feature_extraction.text import CountVectorizer

""" Main """
with open('params.json') as f:
    pars = json.load(f)
    years = tuple(map(int, pars['YEARS'].split('-')))
    max_num_docs = pars['MAX_DOCS']
    text_csv_file = pars['DBLP_CSV_FILE']

folder = 'input/npz'
filenames = [f'{folder}/dblp_{y}.npz' for y in range(*years)]
data = np.concatenate([np.load(n)['data_array'] for n in filenames])
if len(data) > max_num_docs:
    data = data[:max_num_docs, :]
num_docs = len(data)

print('finding nearest neighbours')
model = NearestNeighbors(n_neighbors=100, n_jobs=-1)
model.fit(data)
cols = model.kneighbors(data, return_distance=False)

# adjacency marix
print('creating adjacency matrix')
shape = num_docs, num_docs
A = lil_matrix(shape, dtype=float)
for m in tqdm(range(num_docs)):
    for n in cols[m]:
        A[m, n] = 1 - cosine(data[m], data[n])

# cluster documents
model = Leiden(resolution=1)
cluster_label = model.fit_predict(A.tocsr())
num_clusters = len(set(cluster_label))
print('Created', num_clusters, 'clusters')

# characterize clusters by lexical content
df = pd.read_csv(text_csv_file)
docs = df[df.YEAR.isin(range(*years))].TITLE.to_numpy()[:num_docs]
token_pattern = r'(?:\w+-)?[a-zA-Z]+(?:-\w+)*'
vectorizer = CountVectorizer(
    token_pattern=token_pattern,
    stop_words='english',
    max_df=0.1,
    min_df=5,
    max_features=20000,
    ngram_range=(1, 3))

term_frequency = vectorizer.fit_transform(docs)

terms = vectorizer.get_feature_names_out()
print(len(terms), 'terms counted')

# compute cluster frequency using projection matrix
values = np.ones(num_docs)
cluster_matrix = csr_matrix((values, (cluster_label, range(num_docs))))
term_frequency_per_cluster = cluster_matrix * term_frequency
av_tf = np.asarray(term_frequency_per_cluster.mean(axis=0)).reshape(-1)

# compute chi2 score of every term
print('calculating scores')
columns = ['CLUSTER', 'TERM', 'SCORE']
res = pd.DataFrame(columns=columns)
for m in tqdm(range(num_clusters)):
    scores = list()
    for n, term in enumerate(terms):
        observed = term_frequency_per_cluster[m, n]
        expected = av_tf[n]
        score = (observed - expected) * (observed - expected) / expected
        scores.append((m, term, score))
    cluster_res = pd.DataFrame(scores, columns=columns)
    cluster_res.sort_values('SCORE', ascending=False, inplace=True)
    res = pd.concat([res, cluster_res.head(100)])

output_filename = f'output/cluster_terms_{years[0]}-{years[1]}.xlsx'
with pd.ExcelWriter(output_filename) as writer:
    for m in range(num_clusters):
        sel = res[res.CLUSTER == m].set_index('TERM')
        sel['SCORE'].to_excel(writer, sheet_name=f'cluster {m}')
print('Saved to', output_filename)
