import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
import transformers
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, DistilBertModel
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

import time
import datetime
import random
import os
import sys
import json
import pickle
import pathlib
import re


# path2 = pathlib.Path(__file__).parent.absolute()
PROJECT_ROOT_DIR =  '' # os.path.abspath(os.getcwd())
RETIEVED_DOCUMENTS_DIR = PROJECT_ROOT_DIR + 'retrieved_documents/row-data/'
EXTRACTED_DOCUMENTS_DIR = PROJECT_ROOT_DIR + 'retrieved_documents/extracted-data/'

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

try:
	import en_core_web_sm
	nlp = en_core_web_sm.load()
except:
	import spacy
	nlp = spacy.load("en_core_web_sm")

TAG_RE = re.compile(r'<[^>]+>')
