import os

PROJECT_ROOT_DIR = os.path.abspath(os.getcwd())
print(PROJECT_ROOT_DIR)
DOCUMENTS_DIR = PROJECT_ROOT_DIR + '/retrieved_documents/'

if not os.path.exists(DOCUMENTS_DIR):
    os.mkdir(DOCUMENTS_DIR)

if not os.path.exists(DOCUMENTS_DIR + 'row-data/'):
    os.mkdir(DOCUMENTS_DIR + 'row-data/')

if not os.path.exists(DOCUMENTS_DIR + 'extracted-data/'):
    os.mkdir(DOCUMENTS_DIR + 'extracted-data/')

