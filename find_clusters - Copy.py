# -*- coding: utf-8 -*-
"""
Created on Thu Jul  3 16:47:15 2025

@author: carrraf
"""
import json
import os
import pandas as pd
import numpy as np
# import multiprocessing as mp

from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from sknetwork.clustering import Leiden
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

""" Main """
with open('params.json') as f:
    pars = json.load(f)
    input_folder = pars['EMBEDDINGS_FOLDER']
    train_years = pars['TRAINING_SET']
    test_years = pars['TEST_SET']
    max_num_docs = pars['MAX_DOCS']
    n_neighbors = pars['N_NEIGHBORS']
    text_csv_file = pars['DBLP_CSV_FILE']

input_filenames = [os.path.join(input_folder, n)
                   for n in os.listdir(input_folder)
                   if any(str(s) in n for s in train_years)]
data = np.concatenate([np.load(k)['array'] for k in input_filenames])
print('loaded', len(data), 'titles')

if len(data) > max_num_docs:
    selection = np.random.choice(len(data), max_num_docs, replace=False)
    data = data[selection, :]
else:
    selection = None

num_docs = len(data)
1/0
print('finding nearest neighbours')
model = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=-1)
model.fit(data)
cols = model.kneighbors(data, return_distance=False)

# adjacency marix
print('creating adjacency matrix')
values = list()
for m in tqdm(range(num_docs)):
    values.extend(cosine_similarity(data[m].reshape(1, -1), data[cols[m]])[0])
row_ind = np.repeat(range(num_docs), n_neighbors)
col_ind = cols.flatten()
shape = num_docs, num_docs
A = csr_matrix((values, (row_ind, col_ind)), shape=shape)


# cluster documents
model = Leiden(resolution=1)
cluster_label = model.fit_predict(A.tocsr())
num_clusters = len(set(cluster_label))
print('Created', num_clusters, 'clusters')

# characterize clusters by lexical content
df = pd.read_csv(text_csv_file)
if selection is None:
    docs = df[df.YEAR.isin(train_years)].TITLE.to_numpy()
else:
    docs = df[df.YEAR.isin(train_years)].iloc[selection].TITLE.to_numpy()
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
    res = pd.concat([res, cluster_res.head(20)])

timestamp = pd.Timestamp.now().strftime('%Y-%m-%d_%H%M')
output_filename = f'output/clusters_{timestamp}.xlsx'
with pd.ExcelWriter(output_filename) as writer:
    pd.DataFrame(pars).transpose().to_excel(writer, sheet_name='pars')
    for m in range(num_clusters):
        sel = res[res.CLUSTER == m].set_index('TERM')
        sel['SCORE'].to_excel(writer, sheet_name=f'cluster {m}')
print('Saved to', output_filename)
