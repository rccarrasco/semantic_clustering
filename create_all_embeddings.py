# -*- coding: utf-8 -*-
"""
Created on Thu Jul  3 16:47:15 2025

@author: carrraf
"""

import os
import json
import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer
from tqdm import tqdm

if __name__ == '__main__':
    with open('params.json') as f:
        pars = json.load(f)
        input_file = pars['DBLP_CSV_FILE']
        output_file = pars['DBLP_EMBEDDINGS_FILE']
        chunk_size = pars['CHUNK_SIZE']

    with open('H:CONFIG/credentials.json') as f:
        proxies = json.load(f)['proxies']
        os.environ['HTTP_PROXY'] = proxies['http']
        os.environ['HTTPS_PROXY'] = proxies['https']

    df = pd.read_csv(input_file)
    print('preparing model')
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print('calculating embeddings')
    items = df.TITLE.fillna('').to_numpy()
    _res_ = list()
    for n in tqdm(range(0, len(items), chunk_size)):
        _res_.append(model.encode(items[n:n + chunk_size]))
    res = np.concatenate(_res_)
    np.savez_compressed(output_file, array=res)
