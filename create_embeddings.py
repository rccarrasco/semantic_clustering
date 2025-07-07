# -*- coding: utf-8 -*-
"""
Created on Thu Jul  3 16:47:15 2025

@author: carrraf
"""
import gzip
import os
import json
import numpy as np

from sentence_transformers import SentenceTransformer

with open('H:CONFIG/credentials.json') as f:
    proxies = json.load(f)['proxies']
    os.environ['HTTP_PROXY'] = proxies['http']
    os.environ['HTTPS_PROXY'] = proxies['https']

with gzip.open('input/titles.txt.gz') as f:
    docs = [line.decode() for line in f.readlines()]

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(docs[:100000])

output_path = 'input/embeddings.npz'
np.savez_compressed(output_path, data_array=embeddings)
