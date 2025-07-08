#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  6 10:33:12 2025

@author: rafa
"""
import json
import gzip
import xml.sax
import pandas as pd


class Handler(xml.sax.ContentHandler):
    columns = ['TITLE', 'JOURNAL', 'YEAR']

    def __init__(self, output_file):
        self.output_file = output_file
        self.path = list()
        self.text_content = ''
        self.row = dict()
        self.data = list()
        self.num_items = 0

    def startElement(self, name, attrs):
        self.path.append(name)
        if len(self.path) == 2:
            self.num_items += 1

    def endElement(self, name):
        if name == 'article':
            row = tuple(self.row.get(c, '') for c in Handler.columns)
            self.data.append(row)
            print(len(self.data), self.num_items)
        elif name.upper() in Handler.columns:
            self.row[name.upper()] = self.text_content
        elif name == 'dblp':
            self.save()

        self.text_content = ''
        self.path.pop()

    def characters(self, text_content):
        if self.path[-1].upper() in Handler.columns:
            self.text_content += text_content

    def save(self):
        df = pd.DataFrame(self.data, columns=Handler.columns)
        if self.output_file.endswith('gz'):
            df.to_csv(self.output_file, index=False, compression='gzip')
        else:
            df.to_csv(self.output_file, index=False)


if __name__ == "__main__":
    with open('params.json') as f:
        pars = json.load(f)
        input_file = pars['DBLP_XML_FILE']
        output_file = pars['DBLP_CSV_FILE']

    parser = xml.sax.make_parser()
    # parser.setFeature(xml.sax.handler.feature_namespaces, 0)
    handler = Handler(output_file)
    parser.setContentHandler(handler)

    with gzip.open(input_file, mode='rt', encoding='utf-8') as f:
        parser.parse(f)
