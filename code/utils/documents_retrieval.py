from utils.config import *
import trafilatura
import requests
import xml.etree.ElementTree as ET
from boilerpy3 import extractors
import concurrent.futures


from rake_nltk import Rake

rake = Rake()

RETIEVED_DOCUMENTS_DIR = PROJECT_ROOT_DIR + 'retrieved_documents/row-data/'
EXTRACTED_DOCUMENTS_DIR = PROJECT_ROOT_DIR + 'retrieved_documents/extracted-data/'
TOPICS_FILE = PROJECT_ROOT_DIR + '/topics.xml'

extractor = extractors.ArticleExtractor()


def GetTrimmedKeyword(keyword):
    stop_words = ['better', 'difference', 'best']
    trimmed_word = ' '.join([token.lemma_ for token in nlp(keyword) if token.text not in stop_words])
    return trimmed_word

def GetKeywords(text):
    keywords = []
    rake.extract_keywords_from_text(text)
    keywords_rake = rake.get_ranked_phrases()
    for word in keywords_rake:
        trimmed_word = GetTrimmedKeyword(word)
        if trimmed_word: keywords.append(trimmed_word)
    return keywords

def CreatTopicDirs(topic):
    if not os.path.exists(RETIEVED_DOCUMENTS_DIR + 'topic-{}/'.format(topic['id'])):
            os.mkdir(RETIEVED_DOCUMENTS_DIR + 'topic-{}/'.format(topic['id']))
    if not os.path.exists(EXTRACTED_DOCUMENTS_DIR + 'topic-{}/'.format(topic['id'])):
        os.mkdir(EXTRACTED_DOCUMENTS_DIR + 'topic-{}/'.format(topic['id']))

def GetTopics(fileLocarion=TOPICS_FILE):
    topics = []
    tree = ET.parse(fileLocarion)
    root = tree.getroot()
    for element in root:
        topic = {
                'id': element[0].text.strip()
                ,'title': element[1].text.strip()
                }
        topics.append(topic)

    return topics

def GetDocContent(topic_id, uuid, index='cw12'):
    url = uuid
    source_file = trafilatura.fetch_url(url) # g.text

    if not source_file:
        time.sleep(0.5)
        # return ' ', ' '
        return GetDocContent(topic_id, uuid, index)

    data_1 = trafilatura.extract(source_file)
    if data_1:
        data_1 = TAG_RE.sub('', data_1)
        doc_1 = nlp(data_1)
        sents_1 = [sent.text.strip().lower().replace('\n', ' ') for sent in doc_1.sents if len(sent.text) > 20]
    else:
        sents_1 = []

    data_2 = extractor.get_content(source_file)
    if data_2:
        data_2 = TAG_RE.sub('', data_2)
        doc_2 = nlp(data_2)
        sents_2 = [sent.text.strip().lower().replace('\n', ' ') for sent in doc_2.sents if len(sent.text) > 20]
    else:
        sents_2 = []

    final_data = list(set(sents_1) | set(sents_2))
    main_content = '\n'.join(final_data)

    return source_file, main_content

def GetDocumentsTopic(topic, documents,size=60):
    result_obj = {'topic' : topic, 'documents' : []}
    CreatTopicDirs(topic)

    keywords = GetKeywords(topic['title'])

    for doc in documents:
        # print(doc)
        source_file_path = RETIEVED_DOCUMENTS_DIR + 'topic-{}/{}.html'.format(topic['id'], doc['_id'])
        content_file_path = EXTRACTED_DOCUMENTS_DIR + 'topic-{}/{}.txt'.format(topic['id'], doc['_id'])

        doc['source_file_path'] = os.path.abspath(source_file_path)
        doc['content_file_path'] = os.path.abspath(content_file_path)
        result_obj['documents'].append(doc)

        if os.path.exists(source_file_path) and os.path.exists(content_file_path):
            continue
        
        source_file, main_content = GetDocContent(topic['id'], doc['_source']['uuid'])
        with open(source_file_path, 'w', encoding='utf-8') as f:
            f.write(source_file)
        with open(content_file_path, 'w', encoding='utf-8') as f:
            f.write(main_content)

    return result_obj

def DownloadRelatedDocuments():
    retrieval_results = []
    retrieved_documents = pd.read_json("../data/elastic_search_results.json")
    for idx, result in retrieved_documents.iterrows():
        topic = result["topic"]
        documents = result["documents"]
        topic_docs = GetDocumentsTopic(topic, documents)
        retrieval_results.append(topic_docs)
    with open(PROJECT_ROOT_DIR + '../data/retrieval_results.json', 'w', encoding='utf-8') as f:
        json.dump(retrieval_results, f)

def GetRetrievalResults():
    with open(PROJECT_ROOT_DIR + '../data/retrieval_results.json', 'r', encoding='utf-8') as f:
        retrieval_results = json.load(f)
    return retrieval_results