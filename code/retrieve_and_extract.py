from utils.config import *
from utils.model_handler import LoadBertModel, Predict, TextToSentences, EmbedDocument
from utils.documents_retrieval import DownloadRelatedDocuments, GetRetrievalResults
import pandas as pd

global model
bert_model_path = PROJECT_ROOT_DIR + 'models/stack_argument_weighted_model.pt'
model = LoadBertModel(bert_model_path)
DownloadRelatedDocuments()
retrieval_info = GetRetrievalResults()

for ret_obj in retrieval_info:
    for doc in ret_obj['documents']:
        paragraphs = []
        with open(doc['content_file_path'], 'r', encoding='utf-8') as f:
            sentences = f.readlines()
        if not sentences or 1 == len(sentences):
            continue
        args_file_path = EXTRACTED_DOCUMENTS_DIR + 'topic-{}/{}.args'.format(ret_obj['topic']['id'], doc['_id'])
        print(args_file_path)
        doc['args_file_path'] = os.path.abspath(args_file_path)
        if os.path.exists(args_file_path):
            continue
        predicted = Predict(model, sentences)
        args = [sentences[idx].replace('  ', ' ') for idx, y in enumerate(predicted) if 0 != int(y)]
   
        with open(args_file_path, 'w', encoding='utf-8') as f:
            f.write(''.join(args))
            
with open(PROJECT_ROOT_DIR + '../data/retrieval_results_with_args.json', 'w', encoding='utf-8') as f:
    json.dump(retrieval_info, f)
