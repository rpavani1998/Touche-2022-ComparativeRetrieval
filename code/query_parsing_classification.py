from simpletransformers.ner import NERModel
import pandas as pd
import numpy as np
import torch
from scipy.special import softmax
import argparse
from pathlib import Path
import xml.etree.ElementTree as ET
import pandas as pd
import re
from itertools import groupby
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
stopwords = set(stopwords.words('english'))
import sys
import os

def remove_stopwords(data):
    output_array=[]
    for sentence in data:
        temp_list=[]
        for word in sentence.split():
            if word.lower() not in stopwords:
                temp_list.append(word)
        output_array.append(' '.join(temp_list))
    return output_array

args = {"overwrite_output_dir": True, "num_train_epochs": 10, "fp16": False, "train_batch_size": 8, "gradient_accumulation_steps": 4, "evaluate_during_training": False, "learning_rate": 3e-5, "labels_list": ["O", "OBJ", "ASP", "PRED"], "reprocess_input_data": True, "output_dir": "OUTPUT_PATH", "max_seq_length": 64, "use_early_stopping": True}

def query_parse_train_model(model_config):
    train_df = pd.read_csv('data/full.tsv', sep='\t', encoding='utf-8')
    model = NERModel(model_config[0], model_config[1], use_cuda=torch.cuda.is_available(), cuda_device=4)
    train_data_split = train_df
    model.train_model(train_data_split, eval_df=train_data_split, args=args)
    return model 
    
      
def query_parse_predict(model, query_df):
    predict_data_split = query_df    
    to_predict = list()
    for i in sorted(list(set(predict_data_split.sentence_id.tolist()))):
        question = list()
        for _, rows in predict_data_split.iterrows():
            if rows.sentence_id == i:
                question.append(rows.words)
        question = [q for q in question if isinstance(q, str)]
        to_predict.append(' '.join(question))
    predictions, raw_outputs = model.predict(to_predict)
    preds = []
    index = 0
    ids = sorted(list(set(predict_data_split.sentence_id.tolist())))
    results = list()
    for _, (preds, outs) in enumerate(zip(predictions, raw_outputs)):
        for pred, out in zip(preds, outs):
            key = list(pred.keys())[0]
            new_out = out[key]
            preds = list(softmax(np.mean(new_out, axis=0)))
            result = list()
            result.append(ids[index])
            result.append(key)
            result.append(pred[key])
            result.append(preds[np.argmax(preds)])
            results.append(result)
        index += 1

    df_out = pd.DataFrame(results, columns=["sentence_id", "words", "labels", "prob"])
    df_out.to_csv('data/queries_labelled.tsv', sep='\t', index=False)

    
def process_topics_xml_file(file_path):
    mytree = ET.parse(file_path)
    myroot = mytree.getroot()
    queries = []
    for i, item in enumerate(myroot):
        for x in item:        
            if x.tag == "title":
                 queries.append(re.sub(r'[^\w\s]','',(x.text.lower())))
    queries_rm_stopwords = remove_stopwords(queries)
    queries = pd.DataFrame(queries, columns=['Queries'])
    query = queries['Queries'].str.split(expand=True).stack()
    queries = pd.DataFrame({
        'sentence_id': query.index.get_level_values(0) + 1, 
        'words': query.values,
        'labels': ''
    })
    return queries

for model in [["roberta", "roberta-large"]]:
    for lr in [3e-5]:
        args["learning_rate"] = lr
        query_df = process_topics_xml_file(sys.argv[1])
        query_parse_predict(query_parse_train_model(model), query_df)


        
