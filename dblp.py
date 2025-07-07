#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  6 10:33:12 2025

@author: rafa
"""
import gzip
import xml.sax

class DBLP_Handler(xml.sax.ContentHandler):
    def __init__(self):
        self.listen = False # if True, activate title search
        self.title =  False # if True, store text content
        self.content = ''   # text content
        self.titles = list()
        self.depth = 0
        self.progress = 0
        
    def startElement(self, name, attrs):
        if name == 'article':  
            self.listen = True
        elif name == 'title':  
            self.title = True
            self.content = ''
        self.depth += 1
        
    def endElement(self, name):
        if name == 'article':
            title = self.content.strip()
            if len(title) > 0:
                self.titles.append(title)
                # print(title)
            self.listen = False 
        elif name == 'title': 
            self.title = False  
        elif name == 'dblp': # save all titles
            with gzip.open('input/titles.txt.gz', 'wt') as f:
                f.write('\n'.join(self.titles))
        
        self.depth -= 1
        if self.depth == 1:
            self.progress += 1
            print(self.progress)
        
    def characters(self, content):
        if self.listen and self.title:
            self.content += content


if __name__ == "__main__":
    parser = xml.sax.make_parser()
    parser.setFeature(xml.sax.handler.feature_namespaces, 0)
    handler = DBLP_Handler()
    parser.setContentHandler(handler)

    path = 'input/dblp.xml.gz'
    with gzip.open(path, mode='rt', encoding='utf-8') as f:
        parser.parse(f)

