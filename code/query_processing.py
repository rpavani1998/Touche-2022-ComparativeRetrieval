import xml.etree.ElementTree as ET
import pandas as pd
import re
from itertools import groupby
import nltk
from itertools import chain
import itertools
import sys
from nltk.corpus import wordnet
from nltk.corpus import stopwords
nltk.download('omw-1.4')
nltk.download('stopwords')
stopwords = set(stopwords.words('english'))

def remove_stopwords(data):
    output_array=[]
    for sentence in data:
        temp_list=[]
        for word in sentence.split():
            if word.lower() not in stopwords:
                temp_list.append(word)
        output_array.append(' '.join(temp_list))
    return output_array

def synonyms(word):
    synonyms = wordnet.synsets(word)
    lemmas = set(chain.from_iterable([word.lemma_names() for word in synonyms]))
    return list(lemmas)

mytree = ET.parse(sys.argv[1])
myroot = mytree.getroot()
queries = []
query_ids = []
for i, item in enumerate(myroot):
    for x in item:        
        if x.tag == "title":
            queries.append(re.sub(r'[^\w\s]','',(x.text.lower())))
        if x.tag == "number":
            query_ids.append(x.text)

queries_rm_stopwords = remove_stopwords(queries)


query_df = pd.read_csv('data/queries_labelled.tsv', sep='\t')
query_df  = query_df.dropna(how='all', axis='columns')
query_df  = query_df.groupby('sentence_id').agg(lambda x: x.tolist())

objects_, predict, aspects = [], [], []
objects_count = []
for i, query in query_df.iterrows():
    objects = []
    label_occurance = [(label, sum(1 for _ in group)) for label , group in groupby(query['labels'])]
    predict.append(' '.join([query['words'][i] for i, label in enumerate(query['labels']) if label == 'PRED']))
    aspects.append(' '.join([query['words'][i] for i, label in enumerate(query['labels']) if label == 'ASP']))
    index = 0
    for label, count in label_occurance:
        if label == 'OBJ':
           objects.append([' '.join(query['words'][index:index+count])])
        index += count
    objects_count.append(len(objects))
    objects_.append(objects)
                 
query_df['object_1'] = [obj[0][0] for obj in objects_]
query_df['object_2'] = [obj[1][0] for obj in objects_]
query_df['Predict'] = predict
query_df['Ascpect'] = aspects
query_df['Object_Count'] = objects_count
del query_df['prob']

query_tokens = []
predict_syn = []
aspect_syn = []
for _, query in query_df.iterrows():
    query_build_tokens = []
    label_occurance = [(label, sum(1 for _ in group)) for label , group in groupby(query['labels'])]
    index = 0
    predict_synonyms = []
    aspect_synonyms = []
    for label, count in label_occurance:
        if label == 'OBJ':
            query_build_tokens.append([' '.join(query['words'][index:index+count])])
        elif label == 'PRED':
            synonym_words = synonyms(' '.join(query['words'][index:index+count]))
            predict_synonyms.append(synonym_words)
            if len(synonym_words) == 0:
                for word in query['words'][index:index+count]:
                    if word not in stopwords:
                         query_build_tokens.append(synonyms(word))
            else:
                query_build_tokens.append(synonym_words)
        elif label == 'ASP':
            query_build_tokens.append([' '.join(query['words'][index:index+count])])
            aspect_synonyms.append( synonyms(' '.join(query['words'][index:index+count])))
        index += count
    query_tokens.append(query_build_tokens)
    predict_syn.append(predict_synonyms[0] if len(predict_synonyms) == 1 else [])
    aspect_syn.append(aspect_synonyms[0] if len(aspect_synonyms) == 1 else [])

queries = []
for query in query_tokens:
    queries_lst = [list(query_) for query_ in list(itertools.product(*query))]
    queries.append(queries_lst)

data = { 'Topic Id' : query_ids, 'Query': queries_rm_stopwords, 'Expanded Queries': queries, 'object_1': query_df['object_1'], 'object_2': query_df['object_2'], 'Predict' : predict, 'Aspects' : aspects, 'Predict_syn' : predict_syn, 'Aspect_syn' : aspect_syn}

query_df = pd.DataFrame(data)
query_df.to_json('data/expanded_queries_label_based.json')