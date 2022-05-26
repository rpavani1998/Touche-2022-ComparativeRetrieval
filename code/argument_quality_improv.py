from tqdm import tqdm
from debater_python_api.api.debater_api import DebaterApi
from utils.config import PROJECT_ROOT_DIR
import time
from utils.documents_retrieval import GetTopics
import pandas as pd
import sys
import json
import os

debater_api = DebaterApi('0abeffa5335cc942fc7c43e75d41fe33L05')
argument_quality_client = debater_api.get_argument_quality_client()

topics = GetTopics(sys.argv[1])
documents = pd.read_json(PROJECT_ROOT_DIR + '../data/retrieval_results_with_args.json', encoding='utf-8')

topics = []
documents_ = []
for idx, topic_docs in documents.iterrows():
    docs = []
    if topic_docs['documents'] != []:
        for doc in topic_docs['documents']:
            args = []
            with open(doc['args_file_path'], 'r', encoding='utf-8') as f:
                args = f.readlines()
            args_file_path_ = doc['args_file_path'].replace('.args','_.args')
            if os.path.exists(args_file_path_):
                with open(args_file_path_, 'r', encoding='utf-8') as f:
                    args_improv = f.read()
                doc['answer'] = args_improv
            else:
                sentence_topic_dicts = [{'sentence' : sentence, 'topic' : topic_docs['topic']['title']} for sentence in args]
                scores = argument_quality_client.run(sentence_topic_dicts)
                arguments_score = {args[i].replace('\n',''):score for i, score in enumerate(scores)}
                arguments_score = dict(sorted(arguments_score.items(), key=lambda x: x[1], reverse=True))
                doc['answer'] = ''.join(list(arguments_score.keys())[:15])
                with open(args_file_path_, 'w', encoding='utf-8') as f:
                    f.write(''.join(doc['answer']))
            docs.append(doc)
    documents_.append(docs)
    topics.append(topic_docs['topic'])
            
results = []           
for i, doc in enumerate(documents_):
    row = {'topic' : topics[i], 'documents' : documents_[i]}
    results.append(row)
    
with open(PROJECT_ROOT_DIR + '../data/retrieval_results_with_args_.json', 'w', encoding='utf-8') as f:
    json.dump(results, f)
        
