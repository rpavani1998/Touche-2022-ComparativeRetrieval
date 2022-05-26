from utils.config import *

batch_size = 32
max_seq_len = 128

class StackModel(object):
    """docstring for StackModel"""
    def __init__(self, models_dir):
        bert_model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels = 2,
                                                            output_attentions = False, output_hidden_states = False)
        bert_model = bert_model.to(device)
        bert_model.load_state_dict(torch.load(models_dir + '/stack_distilbert_weighted.pt', map_location=device))

        svm_model = pickle.load(open(models_dir + '/stack_svm_model.plk', 'wb'))
        meta_model = pickle.load(open(models_dir + '/stack_meta_model_5_folds.plk', 'wb'))

        self.bert_model = bert_model
        self.svm_model = svm_model
        self.meta_model = meta_model

    def Predict(self, sentences):
        svm_sentences = GetSvmSentences(sentences)
        
        y_bert_preds = Predict(self.bert_model, sentences, logits_enable=True)


        


def LoadBertModel(model_path):
	# model = BertForSequenceClassification.from_pretrained('bert-base-uncased', config=config)
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels = 2,
                                                            output_attentions = False, output_hidden_states = False)
    model = model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model

#########################################################
# class to transform row texts into dataloader
#########################################################
class RowSentencesHandler():
    def __init__(self):
        # bert tokenizer
        # self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased', do_lower_case=True)
        # roberta tokenizer
        self.tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased', do_lower_case=True)

    def GetDataLoader(self, sentences):
        tokens = self.tokenizer.batch_encode_plus(
            sentences,
            max_length = max_seq_len,
            add_special_tokens = True,
            padding=True,
            truncation=True,
            return_token_type_ids=False
        )

        sequence = torch.tensor(tokens['input_ids']) #.to(device)
        mask = torch.tensor(tokens['attention_mask']) #.to(device)

        # wrap tensors
        data = TensorDataset(sequence, mask)
        # sampler for sampling the data during training
        sampler = SequentialSampler(data)
        # dataLoader for validation set
        dataloader = DataLoader(data, sampler = sampler, batch_size = batch_size)
        return dataloader

#########################################################
# Function to use the model for prediction
#########################################################
def Predict(model, sentences, logits_enable=False):
    model.eval()

    text_handler = RowSentencesHandler()
    dataloader = text_handler.GetDataLoader(sentences)
    prediction_result = np.array([])
    logits_result = np.array([[0,0]])

    for batch in dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        # Unpack the inputs from our dataloader
        input_ids, input_mask = batch

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=input_mask)

        logits = outputs[0]
        logits = logits.detach().cpu().numpy()
        prediction_result = np.append(prediction_result, np.argmax(logits, axis=1).flatten())
        logits_result = np.append(logits_result, logits, axis=0)

    if logits_enable:
        return logits_result[1:].tolist()

    return prediction_result.tolist()

def TextToSentences(text):
    document = nlp(text)
    sentences = [s.text for s in document.sents if len(s) > 20]
    return sentences

def EmbedDocument(text_doc):
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    model = DistilBertModel.from_pretrained('distilbert-base-uncased')
    input_ids = torch.tensor(tokenizer.encode("Hello from the other side")).unsqueeze(0)  # Batch size 1
    outputs = model(input_ids)
    last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
    return last_hidden_states