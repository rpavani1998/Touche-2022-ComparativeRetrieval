from tensorflow import keras
from keras.preprocessing.text import Tokenizer
import re
import nltk
from nltk.corpus import stopwords
import pandas as pd
import warnings




with open('data/expanded_queries_label_based.json') as f:
    queries = json.load(f)
topic_docs = pd.read_json(PROJECT_ROOT_DIR + 'retrieval_results_with_args_.json', encoding='utf-8')


for i, topic in topic_docs.iterrows():
    
answer = []
for idx, topic_docs in documents.iterrows():
    if topic_docs['documents'] != []:
        for doc in topic_docs['documents']:
            args = []

            doc['answer'] = args
            
df = pd.read_json("../data/documents_query.json")

df_len = len(df)

words = []
df1 = df
texts = []
max_text_word_len = 0
for i, document in df.iterrows():
    text = document.answer + ' ' + document['Object1'] 
    text = clean_text(text).replace('\d+', '')
    texts.append(text)
    # max_text_word_len = len(list(set(text.split(' ')))) if len(list(set(text.split(' ')))) > max_text_word_len else max_text_word_len
print(len(texts))  
df['answer'] = texts

texts = []
for i, document in df.iterrows():
    text = document.answer + ' ' + document['Object2']
    text = clean_text(text).replace('\d+', '')
    texts.append(text)
df1['answer'] = texts
df = pd.concat([df, df1])


tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(df['answer'].values)
from keras.preprocessing.sequence import pad_sequences
X = tokenizer.texts_to_sequences(df['answer'].values)
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)

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

results1.to_json('../data/stance_detection.json')