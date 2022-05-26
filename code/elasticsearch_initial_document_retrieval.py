

import requests
import unicodedata
import json
import sys
import nltk
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from boilerpy3 import extractors
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import wordnet
import pandas as pd
from nltk import pos_tag
import pickle
from elasticsearch import Elasticsearch
import sys

with open('data/expanded_queries_label_based.json') as f:
    queries = json.load(f)

topics = []
tree = ET.parse(sys.argv[1])
root = tree.getroot()
for element in root:
    topic = {
            'id': element[0].text.strip()
            ,'title': element[1].text.strip()
            }
    topics.append(topic)
    
queries_df = pd.DataFrame(queries)

elastic_search = Elasticsearch("http://localhost:9200")

corpus = []
for line in open(sys.argv[2], 'r'):
   corpus.append(json.loads(line))

for document in corpus:
    jsonDoc = {
                    'text': document['contents'],
                    'uuid' : document['chatNoirUrl']
            }
    res = elastic_search.index(index="raw_documents", id=document['id'], body=jsonDoc)

objects_1 = []
objects_2 = []
search_results = defaultdict(list)
for i, row in queries_df.iterrows():
    must_words_ = [row['Object1'], row['Object2']]
    should_words = []
    objects_1.append(row['Object1'])
    objects_2.append(row['Object2'])
    should_words.extend(row['Predict'].split(' '))
    should_words.extend(row['Predict_syn'])
    should_words.append(row['Aspects'])
    should_words.extend(row['Aspect_syn'])
    should_words = list(set([word for word in should_words if word != '']))
    must_words_ = list(set([word for word in must_words_ if word != '']))
    must_words = [synonyms(word) for word in must_words_]
    for i, word in enumerate(must_words_):
        must_words[i].append(singularize(word))
    must_words_q = ['({0})'.format(' OR '.join([w for w in word])) for word in must_words if len(word) > 1]
    query = {
        "size": 50,
        "query": {
            "query_string": {
                "query": '({0}) {1} {2}'.format(" AND ".join(must_words_), "AND (" + " OR ".join(should_words) + ")" if len(should_words) > 0 else '', " OR " + " OR ".join(must_words_q)if len(must_words_q) > 0  else '')
            }
        }
    }
    search_results[row['Query']].append(elastic_search.search(index="raw_documents_url", body=query)['hits']['hits'])



results_obj = []
for idx, topic in enumerate(topics):
    topic['object_1'] = queries_df[queries_df['Topic Id'] == topic['id']]['Object1']
    topic['object_2'] = queries_df[queries_df['Topic Id'] == topic['id']]['Object2']
    result_obj = {'topic' : topic, 'documents' : []}
    result_obj['documents'] = search_results[list(elastic_search_result.keys())[idx]][0]
    results_obj.append(result_obj)
    
results_obj = pd.DataFrame(results_obj)    
with open('data/elastic_search_results.json', 'w') as f:
    json.dump(results_obj, f)