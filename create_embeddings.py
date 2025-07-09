# -*- coding: utf-8 -*-
"""
Created on Thu Jul  3 16:47:15 2025

@author: carrraf
"""

import os
import json
import pandas as pd
import numpy as np
import multiprocessing as mp

from sentence_transformers import SentenceTransformer
from tqdm import tqdm

if __name__ == '__main__':
    with open('params.json') as f:
        pars = json.load(f)
        years = tuple(map(int, pars['YEARS'].split('-')))
        input_file = pars['DBLP_CSV_FILE']
        output_file = pars['DBLP_EMBEDDINGS_FILE']
        multiproc = pars['MULTIPROCESS']

    with open('H:CONFIG/credentials.json') as f:
        proxies = json.load(f)['proxies']
        os.environ['HTTP_PROXY'] = proxies['http']
        os.environ['HTTPS_PROXY'] = proxies['https']

    df = pd.read_csv(input_file)
    print('preparing model')
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print('calculating embeddings')
    """
    # does not accelerate processing
    devices = ['cpu'] * 12
    pool = model.start_multi_process_pool(devices)
    embeddings = model.encode(docs, pool=pool)
    model.stop_multi_process_pool()
    """
    for year in range(*years):
        print(year)
        items = df[df.YEAR == year].TITLE.fillna('').to_numpy()
        if multiproc:
            num_cpu = mp.cpu_count()
            with mp.Pool(num_cpu - 2) as p:
                res = list(tqdm(p.imap(model.encode, items), total=len(items)))
        else:
            res = model.encode(items)

    head, ext = os.path.splitext(output_file)
    output_path = f'{head}_{year}{ext}'
    np.savez_compressed(output_path, data_array=res)
