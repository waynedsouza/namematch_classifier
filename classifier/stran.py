from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer , util
import sys
from transformers import BertTokenizer , BertForSequenceClassification, BertConfig
from wbertclassifier import WBertClassifier
from transformers import Trainer, TrainingArguments
import numpy as np
from wtrainer import WTrainer
from wdata import Wdata
from wtok import Wtokenize
from sklearn.metrics import accuracy_score, f1_score , confusion_matrix , precision_recall_fscore_support , precision_score , recall_score , precision_recall_curve
import pandas as pd
import re
#import stanza
#en = stanza.download('en')
#en = stanza.Pipeline(lang='en')

#from utils.character_cnn import CharacterIndexer
##DOWNLOAD CAHRACTERMODEL
#sys.path.insert(0, '../characterbert/character_bert/')
from MISC.scripts.modeling.wayne_character_bert import CharacterBertModel
from MISC.scripts.utils.character_cnn import CharacterIndexer



def compute_metrics(pred):
		is_regression=False    
		print("Compute Metrics\n")
		logits, labels = pred
		print("logits1" ,logits )
		print("labels1" , labels)
		logits = logits[0] if isinstance(logits , tuple) else logits
		#preds2 = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
		#print("preds2" , preds2)
		#preds2 = np.squeeze(preds2) if is_regression else np.argmax(preds2, axis=1)
		#print("preds2" , preds2)
		print("logits" ,logits)
		logits= logits.argmax(axis=-1)
		print("CHECK logits" , logits)
		print("CHECK2 ", np.argmax(pred.predictions[0] , axis = -1))

		#print("\nprint(pred.predictions\n",pred.predictions)
		#print("\nprint(pred.label_ids\n",pred.label_ids)
		#print(pred.predictions.shape)
		#print(pred.label_ids.shape)
	
		#labels = pred.label_ids
		#preds = pred.predictions.argmax(-1)
		labels= labels.argmax(axis=-1)
		print("labels" , labels)
		
		f1 = f1_score(labels, logits, average="weighted")
		acc = accuracy_score(labels, logits)
		_precision_score = precision_score(labels, logits)
		_recall_score = recall_score(labels, logits)
		_confusion = confusion_matrix(labels, logits)
		print({"mse": ((logits - labels) ** 2).mean().item()})
		print({"accuracy": acc, "f1": f1 ,"precision" :_precision_score , "recall":_recall_score , 'CM':_confusion })
		"""print({"accuracy": type(acc), "f1": type(f1) ,"precision" :type(_precision_score) , "recall":type(_recall_score) 
		, 'CM':type(_confusion) ,"mse": ((logits - labels) ** 2).mean().item()}})"""
		print(dir(acc))
		print()
		print(dir(_confusion))
		"""
		acc=acc.item()
		f1=f1.item()
		_precision_score=_precision_score.item()
		_recall_score=_recall_score.item()"""
		_confusion=_confusion.tolist()
		return {"accuracy": acc, "f1": f1 ,"precision" :_precision_score , "recall":_recall_score , 'CM':_confusion }

def MClabel(_label):
	if _label ==1:
		return [0,1]
	return [1,0]
#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
		token_embeddings = model_output[0] #First element of model_output contains all token embeddings
		input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
		return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# Sentences we want sentence embeddings for
sentences = ['This is an example sentence', 'Each sentence is converted']
sentences2 = ['Joe frazier is a boxing coach', 'John Travolata is a dancer']

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-roberta-large-v1')
"""

model = AutoModel.from_pretrained('sentence-transformers/all-roberta-large-v1')
"""


config = BertConfig.from_pretrained('bert-base-uncased', num_labels=2)
#model = BertForSequenceClassification(config=config)
# Tokenize sentences
#encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

# Compute token embeddings
"""
with torch.no_grad():
		model_output = model(**encoded_input)"""

# Perform pooling
#sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

# Normalize embeddings
#sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

#print("Sentence embeddings:")
#print(sentence_embeddings)
#util.pytorch_cos_sim(embedding_1, embedding_2)
device='cpu'


#character_bert_model = CharacterBertModel.from_pretrained( '../characterbert/pretrained-models/general_character_bert/')
character_bert_model = CharacterBertModel.from_pretrained( 'MISC/character_model/general_character_bert/')
"""bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

tokenized_text0 = bert_tokenizer.basic_tokenizer.tokenize(sentences[0])
t_lhs = [bert_tokenizer.basic_tokenizer.tokenize(i) for i in sentences]
t_rhs = [bert_tokenizer.basic_tokenizer.tokenize(i) for i in sentences2]
tokenized_text1 = bert_tokenizer.basic_tokenizer.tokenize(sentences[1])
tokenized_text0 = ['[CLS]', *tokenized_text0, '[SEP]']
tokenized_text1 = ['[CLS]', *tokenized_text1, '[SEP]']
t_lhs = [['[CLS]', *i, '[SEP]'] for i in t_lhs]
t_rhs = [['[CLS]', *i, '[SEP]'] for i in t_rhs]
indexer = CharacterIndexer()
input_tensor0 = indexer.as_padded_tensor([tokenized_text0])
input_tensor1 = indexer.as_padded_tensor([tokenized_text1])
input_tensor_lhs = indexer.as_padded_tensor(t_lhs).to(device)
input_tensor_rhs = indexer.as_padded_tensor(t_rhs).to(device)"""
#model=character_bert_model
"""model = CharacterBertModel.from_pretrained(
		'pretrained-models/general_character_bert/')"""
"""
a = model(input_tensor0)
b = model(input_tensor1)
o =torch.cat((a[1],b[1]),dim=-1)"""



#from fps import estimate_tokens , floating_point_ops
#config = BertConfig.from_pretrained('bert-base-uncased', num_labels=2)  # binary classification
#in_features=1536
#out_features=768
#bert_model = BertForSequenceClassification(config=config)
#bert_model.bert = character_bert_model
wbert_model=WBertClassifier(config)
wbert_model.bert = character_bert_model
#wbert_model.estimate_tokens = estimate_tokens
#wbert_model.floating_point_ops=floating_point_ops
#setattr(wbert_model , 'estimate_tokens' , estimate_tokens)
#setattr(wbert_model , 'floating_point_ops' , floating_point_ops)


"""
a= bert_model(input_tensor0)
b= bert_model(input_tensor1)
adapt = nn.Linear(in_features, out_features,device='cpu')
"""





batch_size = 64
model_ckpt='blah'
logging_steps = 100 #len(dataset_local_encoded['train'])// batch_size
model_name = f"{model_ckpt}-gc-indep"
training_args = TrainingArguments(output_dir=model_name,
																	num_train_epochs=1,
																	learning_rate=2e-5,
																	per_device_train_batch_size=batch_size,
																	per_device_eval_batch_size=batch_size,
																	weight_decay=0.001,                               
																	disable_tqdm=False,
																	logging_steps=logging_steps,
																	push_to_hub=False, 
																	log_level="error",
																	save_strategy="steps",
																	save_steps=10,
																	save_total_limit=1,
																	evaluation_strategy="steps",
																	eval_steps=1,
																	load_best_model_at_end =True,
																	metric_for_best_model ='eval_f1'
																	)


def message_length(x):
	# returns total number of characters
	return len(x)

def num_capitals(x):
	_, count = re.subn(r'[A-Z]', '', x) # only works in english
	return count

def num_punctuation(x):
	_, count = re.subn(r'\W', '', x)
	return count
def word_count(x):
	b=re.split(r'[^a-zA-Z0-9]',x.strip())
	return len(list(filter(None,b)))
def word_counts(x, pipeline=None):
	doc = pipeline(x)
	count = sum( [ len(sentence.tokens) for sentence in doc.sentences] )
	return count
def word_counts_v3(x, pipeline=None):
  doc = pipeline(x)
  count = 0
  for sentence in doc.sentences:
	  for token in sentence.tokens:
		  if token.words[0].upos not in ['PUNCT', 'SYM']:
			  count += 1
  return count








sentences = ['This is an example sentence', 'John Travolata is a dancer' , 'Shane Warne was  acricketeer' , 'John lynch was a business man' ,'Jacques Moore was unknown'] 
sentences2 = ['This is another example', 'John Travolata is an actor' ,'Shane Warne was known for his exceptional fielding','John is retired','Roger Moore playedd James Bond'] 
sentences3 = ['TEST_This is an example sentence', 'TEST_John Travolata is a dancer' , 'TEST_Shane Warne was  acricketeer' , 'TEST_John lynch was a business man' ,'TEST_Jacques Moore was unknown'] 
sentences4 = ['TEST_This is another example', 'TEST_John Travolata is an actor' ,'TEST_Shane Warne was known for his exceptional fielding','TEST_John is retired','TEST_Roger Moore playedd James Bond'] 
labels = [0 , 1 , 1 , 1 , 0] 
#labels = [MClabel(i) for i in labels]
#ds_train = Wdata(sentences ,sentences2 , labels )
#ds_test = Wdata(sentences3 ,sentences4 , labels )
ds_train = Wdata()
ds_test = Wdata()
st_model = 'all-roberta-large-v1' #@param ['paraphrase-mpnet-base-v2', 'all-mpnet-base-v1', 'all-mpnet-base-v2', 'stsb-mpnet-base-v2', 'all-MiniLM-L12-v2', 'paraphrase-albert-small-v2', 'all-roberta-large-v1']
st_model='sentence-transformers/' + st_model
tok_model = 'bert-base-uncased'

wtokenizer = Wtokenize.from_pretrained(tok_model)
wtokenizer.addSentenceModel(st_model)
wbert_model.addSentenceModel(st_model)

a=pd.DataFrame(zip(sentences,sentences2,labels))
a.columns=['sentence1' , 'sentence2' , 'label']
a['Capitals1'] = a['sentence1'].apply(num_capitals)
a['Punctuation1'] = a['sentence1'].apply(num_punctuation)
a['Length1'] = a['sentence1'].apply(message_length)
a['Words1'] = a['sentence1'].apply(word_count)
#a['Words1V3'] = a['sentence1'].apply(word_counts_v3)
a['Capitals2'] = a['sentence2'].apply(num_capitals)
a['Punctuation2'] = a['sentence2'].apply(num_punctuation)
a['Length2'] = a['sentence2'].apply(message_length)
a['Words2'] = a['sentence2'].apply(word_count)
#a['Words2V3'] = a['sentence2'].apply(word_counts_v3)
_train = a.sample(frac=0.6)
_test = a.drop(_train.index)
ds_train.from_df(_train)
ds_test.from_df(_test)
trainer = WTrainer(model=wbert_model, args=training_args, 
									compute_metrics=compute_metrics,
									train_dataset=ds_train,
									eval_dataset=ds_test,
									tokenizer=wtokenizer)
"""
trainer = WTrainer(model=wbert_model, args=training_args, 
									compute_metrics=compute_metrics,
									train_dataset=ds,
									eval_dataset=ds,
									tokenizer=wtokenizer)

"""
#trainer.train()

