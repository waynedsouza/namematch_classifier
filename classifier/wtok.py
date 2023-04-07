
from transformers import BertTokenizer,PreTrainedTokenizer
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer, InputExample, losses, models, datasets, evaluation, util,LoggingHandler
import sys
import torch
import os
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union
import pandas as pd
"""sys.path.insert(0, '../characterbert/character_bert/')
sys.path.insert(0, 'characterbert/character_bert/')"""
#sys.path.insert(0, '/characterbert/character_bert/')

try:
	from canadarealestate.phoneclassifier.MISC.scripts.utils.character_cnn import CharacterIndexer
except:
	sys.path.insert(0, 'F:\\Pycode\\canadarealestate\\characterbert\\character_bert\\')
	from MISC.scripts.utils.character_cnn import CharacterIndexer
#from utils.character_cnn import CharacterIndexer
class Wtokenize(BertTokenizer):
	
	def __init__(self , *args , **kwargs):
		super().__init__(
			 *args,**kwargs
		)
		self.sentence_model=None
		self.char_tokenizer = None
		self.sent_tokenizer = None
		#st_model = 'all-roberta-large-v1' #@param ['paraphrase-mpnet-base-v2', 'all-mpnet-base-v1', 'all-mpnet-base-v2', 'stsb-mpnet-base-v2', 'all-MiniLM-L12-v2', 'paraphrase-albert-small-v2', 'all-roberta-large-v1']
	"""
	@classmethod
	def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], *init_inputs, **kwargs):
		print("From pretrained called with clas" , cls ,"\n")
		self.char_tokenizer= super().from_pretrained(cls, pretrained_model_name_or_path, *init_inputs, **kwargs)
		print(self.char_tokenizer ,"self.char_tokenizer\n")
		return self.char_tokenizer"""
	
	def addSentenceModel(self , model_path):
		if type(model_path) is SentenceTransformer:
		  self.sentence_model = model_path
		else:
		  self.sentence_model = SentenceTransformer(model_path)
		#print("addsentencemodel" , model_path)
		#print( self.sentence_model ," self.sentence_model\n")
		#self.sent_tokenizer = AutoTokenizer.from_pretrained(self.sentence_model)

	def __call__(self , *args):
		print("Tokenizer call called " , args)
		raise Exception("Not programmed")
		#print("Tokenizer: " , inputs)
		#print(pad=pad)
		#print("Args", args)
		texts = ['hello world!', 'good day']
		texts2 = [s.split() for s in texts]
		vecs = bc.encode(texts2, is_tokenized=True)
		return vecs
	def tokenize(self , _text):
		print("In tokenize")
	def _tokenize(self, text):
		print("_tok",text)
	def prepare(self , features):
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		sentences ,sentences2 = features
		sent1_params , sent2_params , ip1 , ip2=None , None,None , None
		if type(sentences) is pd.core.frame.DataFrame and type(sentences2) is pd.core.frame.DataFrame and len(sentences.index)== 1 and len(sentences2.index)==1:
			sentences ,*sent1_params = sentences.loc[0].to_list()
			sentences2 ,*sent2_params = sentences2.loc[0].to_list()
			sentences,sentences2=[sentences],[sentences2]
			sent1_params , sent2_params = [[i] for i in sent1_params] , [[i] for i in sent2_params]
		if type(sentences) is pd.core.frame.DataFrame and type(sentences2) is pd.core.frame.DataFrame and len(sentences.index) > 1 and len(sentences2.index)>1:
			sentences ,*sent1_params = list(zip(*sentences.values))
			sentences2 ,*sent2_params = list(zip(*sentences2.values))
			sentences,sentences2=list(sentences),list(sentences2)
			sent1_params , sent2_params = [list(i) for i in sent1_params] , [list(i) for i in sent2_params]
		elif type(sentences[0] ) is tuple and type(sentences2[0] ) is tuple:				 
			sentences ,*sent1_params = list(zip(*sentences))
			sentences2 ,*sent2_params = list(zip(*sentences2))
		#print("__CHECK0__\n" , sentences ,"\n__CHECK0__\n" )
		#print("__CHECK0.1__\n" , sentences2 ,"\n__CHECK0.1__\n" )
		#print("__CHECK0.2__\n" , sent1_params ,"\n__CHECK0.2__\n" )
		#print("__CHECK0.3__\n" , sent2_params ,"\n__CHECK0.3__\n" )
		t_lhs = [self.basic_tokenizer.tokenize(i) for i in sentences]
		t_rhs = [self.basic_tokenizer.tokenize(i) for i in sentences2]
		t_lhs = [['[CLS]', *i, '[SEP]'] for i in t_lhs]
		t_rhs = [['[CLS]', *i, '[SEP]'] for i in t_rhs]
		#print("__CHECK1__\n" , t_lhs ,"\n__CHECK1__\n" )
		#print("__CHECK1.1__\n" , t_rhs ,"\n__CHECK1.1__\n" )
		indexer = CharacterIndexer()
		input_tensor_lhs = indexer.as_padded_tensor(t_lhs).to(device)
		input_tensor_rhs = indexer.as_padded_tensor(t_rhs).to(device)
		#print("__CHECK2__\n" , input_tensor_lhs ,"\n__CHECK2__\n" )
		#print("__CHECK2.1__\n" , input_tensor_rhs ,"\n__CHECK2.1__\n" )
		if self.sentence_model is not None:
			lhs = self.sentence_model.tokenize(sentences)
			rhs = self.sentence_model.tokenize(sentences2)
			ip1,ip2 = {} ,{}
			ip1['input_ids']=lhs['input_ids']
			ip1['attention_mask'] = lhs['attention_mask']

			ip2['input_ids']=rhs['input_ids']
			ip2['attention_mask']= rhs['attention_mask']
			ip1['input_ids'] = ip1['input_ids'].to(device)
			ip2['input_ids'] = ip2['input_ids'].to(device)
			ip1['attention_mask']=ip1['attention_mask'].to(device)
			ip2['attention_mask']=ip2['attention_mask'].to(device)
			#print("__CHECK3__\n" , ip1 ,"\n__CHECK3__\n" )
			#print("__CHECK4__\n" , ip2 ,"\n__CHECK4__\n" )
		if sent1_params is not None:
			sent1_params=torch.Tensor(sent1_params).to(device)
		if sent2_params is not None:
			sent2_params=torch.Tensor(sent2_params).to(device)
		#print("__CHECK5__\n" , sent1_params ,"\n__CHECK5__\n" , sent1_params.shape )
		#print("__CHECK6__\n" , sent2_params ,"\n__CHECK6__\n" ,sent2_params.shape)
		features=input_tensor_lhs,input_tensor_rhs
		#print("__CHECK7__\n" , features ,"\n__CHECK7__\n" )
		return {'input_ids':features,'ip1':ip1 ,'ip2':ip2 ,'sent1_params' :sent1_params , 'sent2_params':sent2_params}
	def pad(self , features , 
			padding,
			max_length,
			pad_to_multiple_of,
			return_tensors):
			device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
			#print(features ,type(features))
			f , labels = list(zip(*features))
			sentences ,sentences2 = zip(*f)
			sent1_params , sent2_params , ip1 , ip2=None , None,None , None
			if type(sentences[0] ) is tuple and type(sentences2[0] ) is tuple:				 
				sentences ,*sent1_params = list(zip(*sentences))
				sentences2 ,*sent2_params = list(zip(*sentences2))
			#bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') # its in self as self is WTokenize derived from BertTokenizer
				
			t_lhs = [self.basic_tokenizer.tokenize(i) for i in sentences]
			t_rhs = [self.basic_tokenizer.tokenize(i) for i in sentences2]
			t_lhs = [['[CLS]', *i, '[SEP]'] for i in t_lhs]
			t_rhs = [['[CLS]', *i, '[SEP]'] for i in t_rhs]
			#print(t_lhs ,"\nt_lhs\n")
			indexer = CharacterIndexer()
			input_tensor_lhs = indexer.as_padded_tensor(t_lhs).to(device)
			input_tensor_rhs = indexer.as_padded_tensor(t_rhs).to(device)
			if self.sentence_model is not None:
				lhs = self.sentence_model.tokenize(sentences)
				rhs = self.sentence_model.tokenize(sentences2)
				ip1,ip2 = {} ,{}
				ip1['input_ids']=lhs['input_ids']
				ip1['attention_mask'] = lhs['attention_mask']

				ip2['input_ids']=rhs['input_ids']
				ip2['attention_mask']= rhs['attention_mask']
				ip1['input_ids'] = ip1['input_ids'].to(device)
				ip2['input_ids'] = ip2['input_ids'].to(device)
				ip1['attention_mask']=ip1['attention_mask'].to(device)
				ip2['attention_mask']=ip2['attention_mask'].to(device)
			if sent1_params is not None:
				sent1_params=torch.Tensor(sent1_params).to(device)
			if sent2_params is not None:
				sent2_params=torch.Tensor(sent2_params).to(device)
			#print("in pad features" ,features)
			#print("Pad called max_length" , max_length)
			#print("Pad called " , padding)
		 
			#print("lpad pad_to_multiple_of called" , pad_to_multiple_of)
			#print("lpad return_tensors called" , return_tensors)
			features=input_tensor_lhs,input_tensor_rhs
			return {'input_ids':features,'labels':torch.Tensor(labels).to(device) ,'ip1':ip1 ,'ip2':ip2 ,'sent1_params' :sent1_params , 'sent2_params':sent2_params}
			
	def pad1(self ,blah , pad ,padding, max_length,pad_to_multiple_of,return_tensors=False, *args):
		print("Pad called " , pad)
		print("Pad called max_length" , max_length)
		print("Pad called " , padding)
		print("lpad called" , args)
		print("lpad pad_to_multiple_of called" , pad_to_multiple_of)
		print("lpad return_tensors called" , return_tensors)