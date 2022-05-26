from utils.config import PROJECT_ROOT_DIR
from whoosh.analysis import StemmingAnalyzer
from rank_bm25 import BM25Okapi
from utils.documents_retrieval import GetTopics
from sentence_transformers import SentenceTransformer, util
import json
import pandas as pd
import sys
import re
import nltk
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
import keras
import warnings
warnings.filterwarnings("ignore")

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))
DOCUMENTS_PER_TOPIC = 25
BASE_DATA_DIR = PROJECT_ROOT_DIR + 'retrieved_documents/'
OUTPUT_FILE_NAME = PROJECT_ROOT_DIR + 'run.txt'
TAG = 'tfid_arg_similarity'
MAX_NB_WORDS = 50000
MAX_SEQUENCE_LENGTH = 500
EMBEDDING_DIM = 100
tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
word_index = tokenizer.word_index
analyzer = StemmingAnalyzer()
embedding_model = SentenceTransformer('paraphrase-distilroberta-base-v1')

def stance_detection(df):
    df_len = len(df)
    words = []
    df1 = df
    texts = []
    max_text_word_len = 0
    for i, document in df.iterrows():
        text = document.answer + ' ' + document['object_1'] 
        text = clean_text(text).replace('\d+', '')
        texts.append(text)
    df['answer'] = texts

    texts = []
    for i, document in df.iterrows():
        text = document.answer + ' ' + document['object_2']
        text = clean_text(text).replace('\d+', '')
        texts.append(text)
    df1['answer'] = texts
    df = pd.concat([df, df1])

    tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
    tokenizer.fit_on_texts(df['answer'].values)
    from tensorflow import keras
    X = tokenizer.texts_to_sequences(df['answer'].values)
    X = keras.preprocessing.sequence.pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
    
    model = keras.models.load_model('models/lst_stance_detection_model')
    predictions = model.predict(X)

    expected = []
    prediction = []
    for i, preds in enumerate(predictions):
        prediction.append([i for i, pred in enumerate(preds) if pred == max(preds)])

    df['Prediction'] = prediction
    results1 = df[:df_len]
    results2 = df[df_len:]
    results1['Obj1_pred'] = results1['Prediction']
    results1['Obj2_pred'] = results2['Prediction']

    stance_label = {0:'NO', 1:'NEUTRAL',2:'FIRST', 3:'SECOND'}

    label_stance_pred = [stance_label[val[0]] for val in results1['Obj1_pred']]
    return label_stance_pred

def clean_text(text):
    text = str(text)
    text = text.lower() 
    text = REPLACE_BY_SPACE_RE.sub(' ', text) 
    text = BAD_SYMBOLS_RE.sub('', text)  
    text = text.replace('x', '')
    text = ' '.join(word for word in text.split() if word not in STOPWORDS)
    return text

def GetTokenizedDocuments(analyzer, documents):
    tokenized_docs = []
    for doc in documents:
        try:
            with open(doc['args_file_path'], 'r', encoding='utf-8') as f:
                args_list = f.readlines()
        except:
            args_list = []
        arg_text = ''.join(args_list)
        tokenized_docs.append([token.text for token in analyzer(arg_text)])
    return tokenized_docs

def arg_tfidf_ranking(query, documents):
    tokenized_query = [token.text for token in analyzer(query)]
    tokenized_docs = GetTokenizedDocuments(analyzer, documents)
    bm25 = BM25Okapi(tokenized_docs)
    doc_scores = bm25.get_scores(tokenized_query)
    for index, doc in enumerate(documents):
        doc['tfidf_score'] = doc_scores[index]

def arg_support_ranking(query, documents):
    for doc in documents:
        try:
            with open(doc['args_file_path'], 'r', encoding='utf-8') as f:
                args_list = f.readlines()
        except:
            args_list = []

        with open(doc['content_file_path'], 'r', encoding='utf-8') as f:
            all_sents_list = f.readlines()

        doc['arg_support_score_sum'] = len(args_list)
        doc['arg_support_score_norm'] = len(args_list) / len(all_sents_list)

def arg_similarity_ranking(query, documents):
    query_embedding = embedding_model.encode(query, convert_to_tensor=True)

    for doc in documents:
        try:
            with open(doc['args_file_path'], 'r', encoding='utf-8') as f:
                args_list = f.readlines()

            arg_embeddings = embedding_model.encode(args_list, convert_to_tensor=True)
            sentences_similarities = util.pytorch_cos_sim(query_embedding, arg_embeddings).flatten().tolist()
            doc['similarity_score_sum'] = sum(sentences_similarities)
            doc['similarity_score_avg'] = sum(sentences_similarities) / len(sentences_similarities)
        except:
            args_list = []
            doc['similarity_score_avg'] = 0.0
            doc['similarity_score_sum'] = 0.0

def normalize_and_compute(documents):
    max_tfidf_score     = max([ doc['tfidf_score'] for doc in documents])
    max_es_score  = max([ doc['_score'] for doc in documents])
    for doc in documents:
        doc['tfidf_score_norm']     = float(doc['tfidf_score']) / max_tfidf_score
        doc['es_score_norm']  = float(doc['_score']) / max_es_score

    tfidf_w         = 25
    arg_support_w   = 25
    similarity_w    = 15

    for doc in documents:
        doc['final_score'] = tfidf_w       * doc['tfidf_score_norm']    \
                            + arg_support_w * doc['arg_support_score_norm'] \
                            + similarity_w  * doc['similarity_score_avg']  
   

def rank_documents(query, documents):
    arg_tfidf_ranking(query, documents)
    arg_support_ranking(query, documents)
    arg_similarity_ranking(query, documents)
    normalize_and_compute(documents)


topics = GetTopics(sys.argv[1])
print(PROJECT_ROOT_DIR + '../data/retrieval_results_with_args_.json')
documents = pd.read_json(PROJECT_ROOT_DIR + '../data/retrieval_results_with_args_.json', encoding='utf-8')

qids = []
docs = []
ranks = []
scores = []
tags = []
output_lines = []
answers = []
object_1 = []
object_2 = []

for idx, topic_docs in documents.iterrows():
    if topic_docs['documents'] != []:
        rank_documents(topic_docs['topic']['title'], topic_docs['documents'])
        ranked_docs = sorted(topic_docs['documents'], key=lambda x: x['final_score'], reverse=True)
        for rank, doc in enumerate(ranked_docs[:DOCUMENTS_PER_TOPIC], start=1):
            qids.append(topic_docs['topic']['id'])
            docs.append(doc['_id'])
            ranks.append(rank)
            scores.append(doc['final_score'])
            tags.append(TAG)
            answers.append(doc['answer'])
            object_1.append(topic_docs['topic']['object_1'])
            object_2.append(topic_docs['topic']['object_2'])

results = {'qid':qids,'doc':docs,'rank':ranks,'score':scores,'tag':tags,'answer':answers,'object_1':object_1, 'object_2':object_2}
df = pd.DataFrame(results)

stances = stance_detection(df)
output_lines = []
for i,row in df.iterrows():
    if row['score'] > 0.0:
        output_line = '{} {} {} {} {} {}\n'.format(row['qid'], stances[i], row['doc'], row['rank'], row['score'], row['tag'])
        output_lines.append(output_line)

with open('../output/run.txt', 'w') as f:
    for line in output_lines:
        f.write(line)