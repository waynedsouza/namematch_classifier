from transformers import BertForSequenceClassification
import torch
from torch import nn
from transformers.modeling_outputs import SequenceClassifierOutput
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
from sentence_transformers import SentenceTransformer, InputExample, losses, models, datasets, evaluation, util,LoggingHandler
DEBUG=False
class WBertClassifier(BertForSequenceClassification):
	is_eval= False
	def __init__(self, config):
		#print("config" , config)
		#input("continue")
		super().__init__(config)
		dropout=0.5
		
		self.sentence_model=None
		num_params = 4
		#config.num_labels
		self.linear = nn.Linear(config.hidden_size*2, 768)
		self.linear2 = nn.Linear(1024*2,768 )
		self.linear3 = nn.Linear(768*2,768*2 )
		self.linear3_p = nn.Linear(768*2+64,768*2 )
		self.linear_params = nn.Linear(num_params*2,64)
		self.classifier = nn.Linear(768*2,config.num_labels )
		self.dropout = nn.Dropout(dropout)
		nn.init.xavier_normal_(self.linear.weight)
		nn.init.xavier_normal_(self.linear2.weight)
		nn.init.xavier_normal_(self.linear3.weight)
		nn.init.xavier_normal_(self.classifier.weight)
		try:
			self.init_weights()
		except Exception as e:
			print("\nWBertClassifier Init Weihghts exc epted out\n" ,e)
		self.post_init()
	"""
	def eval(self):		
		super().eval()
		self.is_eval = True
	def train(self, mode = True):		
		super().train(mode)
		self.is_eval = False
		"""

	def addSentenceModel(self , model_path):
		if type(model_path) is SentenceTransformer:
		  self.sentence_model = model_path
		else:
		  self.sentence_model = SentenceTransformer(model_path)

	def num_parameters(self, only_trainable: bool = False, exclude_embeddings: bool = False) -> int:
		"""
		Get number of (optionally, trainable or non-embeddings) parameters in the module.

		Args:
			only_trainable (`bool`, *optional*, defaults to `False`):
				Whether or not to return only the number of trainable parameters

			exclude_embeddings (`bool`, *optional*, defaults to `False`):
				Whether or not to return only the number of non-embeddings parameters

		Returns:
			`int`: The number of parameters.
		"""

		if exclude_embeddings:
			embedding_param_names = [
				f"{name}.weight" for name, module_type in self.named_modules() if isinstance(module_type, nn.Embedding)
			]
			non_embedding_parameters = [
				parameter for name, parameter in self.named_parameters() if name not in embedding_param_names
			]
			return sum(p.numel() for p in non_embedding_parameters if p.requires_grad or not only_trainable)
		else:
			return sum(p.numel() for p in self.parameters() if p.requires_grad or not only_trainable)

   
	def estimate_tokens(self, input_dict: Dict[str, Union[torch.Tensor, Any]]) -> int:
		"""
		Helper function to estimate the total number of tokens from the model inputs.

		Args:
			inputs (`dict`): The model inputs.

		Returns:
			`int`: The total number of tokens.
		"""
		if not hasattr(self, "warnings_issued"):
			self.warnings_issued = {}
		if self.main_input_name in input_dict:
			_ip1 , _ip2 = input_dict[self.main_input_name]
			#print("_ip1 , _ip2 shape" , _ip1.shape , _ip2.shape)
			#print("_ip1 , _ip2 numel" , _ip1.numel() , _ip2.numel())
			return _ip1.numel() + _ip2.numel()
		elif "estimate_tokens" not in self.warnings_issued:
			logger.warning(
				"Could not estimate the number of tokens of the input, floating-point operations will not be computed"
			)
			self.warnings_issued["estimate_tokens"] = True
		return 0
	
	def floating_point_ops(
			self, input_dict: Dict[str, Union[torch.Tensor, Any]], exclude_embeddings: bool = True
		) -> int:
			"""
			Get number of (optionally, non-embeddings) floating-point operations for the forward and backward passes of a
			batch with this transformer model. Default approximation neglects the quadratic dependency on the number of
			tokens (valid if `12 * d_model << sequence_length`) as laid out in [this
			paper](https://arxiv.org/pdf/2001.08361.pdf) section 2.1. Should be overridden for transformers with parameter
			re-use e.g. Albert or Universal Transformers, or if doing long-range modeling with very high sequence lengths.

			Args:
				batch_size (`int`):
					The batch size for the forward pass.

				sequence_length (`int`):
					The number of tokens in each line of the batch.

				exclude_embeddings (`bool`, *optional*, defaults to `True`):
					Whether or not to count embedding and softmax operations.

			Returns:
				`int`: The number of floating-point operations.
			"""
			

			return 6 * self.estimate_tokens(input_dict) * self.num_parameters(exclude_embeddings=exclude_embeddings)

	def forward(
		self,
		input_ids = None,
		attention_mask= None,
		token_type_ids= None,
		position_ids= None,
		head_mask= None,
		inputs_embeds = None,
		labels= None,
		output_attentions = None,
		output_hidden_states = None,
		return_dict= None,
		ip1=None,
		ip2=None,
		sent1_params=None ,
		sent2_params=None
		):
		return_dict = return_dict if return_dict is not None else self.config.use_return_dict
		#print("FOrward Start" , ip1 , ip2)
		#print("return_dict" , return_dict)
		#print("labels" , labels)
		#print(input_ids , type(input_ids))
		#print("\nwbertClassifier\n")
		#print(input_ids ,"__input_ids")
		#print(type(input_ids) ,"__input_ids")
		#print(len((input_ids)) ,"__input_ids")
		#print(attention_mask ,"__attention_mask")
		#print(ip1 ,"ip1")
		#print(sent1_params ,"sent1_params")
		assert((type(input_ids) is tuple or type(input_ids) is list ) and len(input_ids) ==2)
		#print("\n\nSTART TRAIL\n")
		

		tokenized_text0, tokenized_text1 = input_ids
		if DEBUG:
			print("tokenized_text0\t" , tokenized_text0.shape)
			print("tokenized_text1\t" , tokenized_text1.shape)
		if self.sentence_model is not None and ip1 is not None and ip2 is not None:
			#print("\nIn Wayne \n")
			out1_sm = self.sentence_model(ip1)
			out2_sm = self.sentence_model(ip2)
			#ss1= out1_sm['sentence_embedding'].shape
			if DEBUG:
				print("out1_sm: " ,out1_sm['sentence_embedding'].shape , "\n sentence_embedding shape")
				print("out2_sm: ", out2_sm['sentence_embedding'].shape , "\n sentence_embedding shape")

		
		outs={}
		for idx, i in enumerate(input_ids):
			#print(idx , i ,"\n")
			outs[idx] = self.bert(
				i,
				attention_mask=attention_mask,
				token_type_ids=token_type_ids,
				position_ids=position_ids,
				head_mask=head_mask,
				inputs_embeds=inputs_embeds,
				output_attentions=output_attentions,
				output_hidden_states=output_hidden_states,
				return_dict=return_dict)
		#print("\n Intermediate Op\n",idx , outs[idx][0].shape , outs[idx][1].shape)
		#print(outs[0][1].shape , "BACK IN WBERT bSE\n")
		try:
			pooled_output1 = outs[0][1]
			pooled_output2 = outs[1][1]
		except:
			print("Excepted")
		
		try:
			if DEBUG:
				print(pooled_output1.shape , "WBERT pooledout1 bSE\n")
				print(pooled_output2.shape , "WBERT pooledout2 bSE\n")
				#print(pooled_output1[0] == pooled_output2[0] ,"__ditto check")
		except:
			pass
		
		o =torch.cat((pooled_output1, pooled_output2,),dim=-1)
		if DEBUG:
			print("O shape " , o.shape)
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		bn1 = nn.BatchNorm1d(768*2, affine=False,device=device)
		#print("__pre__\n" , o.shape , "\n__pre\n")
		#print(self.training ,"____self.training\n")
		#if self.training:
		if self.training:
			o = bn1(o)
			o= self.dropout(o)
		o2 =torch.cat((out1_sm['sentence_embedding'], out2_sm['sentence_embedding']),dim=-1)
		if DEBUG:print("O2 shape " , o2.shape)
		bn2 = nn.BatchNorm1d(1024*2, affine=False,device=device)
		if self.training:
			o2= bn2(o2)
			o2= self.dropout(o2)
		#print(o.shape , "HERE ---> WBERT catop bSE\n")
		#print(o2.shape , "HERE2 sent ---> WBERT catop bSE\n")
		logits1 = self.linear(o)
		if DEBUG:print("logits1 shape" , logits1.shape)
		#if logits1.shape[0]>1:
		logits1=self.dropout(logits1)
		logits2 = self.linear2(o2)
		if DEBUG:print("logits2 shape" , logits2.shape)
		#if logits2.shape[0]>1:
		if self.training:
			logits2=self.dropout(logits2)
		if sent1_params is not None and sent2_params is not None:
			t1_params = sent1_params
			t2_params = sent2_params
			if DEBUG:print("t1_params " , t1_params)
			if DEBUG:print("t2_params " , t2_params)
			"""
			if t1_params.shape[0] == 1 and t2_params.shape[0] == 1:
				t3_params = torch.cat((t1_params , t2_params),dim=-1)
			else:
				t3_params = torch.cat((t1_params.t() , t2_params.t()),dim=-1)"""
			t3_params = torch.cat((t1_params.t() , t2_params.t()),dim=-1)
			if DEBUG:print("t3_params " , t3_params)
			if DEBUG:print("t3_params" , t3_params.shape)
			logits_linear = self.linear_params(t3_params)
			#if logits_linear.shape[0] >1:
			if self.training:
				logits_linear=self.dropout(logits_linear)
			if DEBUG:print("t3 logits " , logits_linear.shape)
			o3 =torch.cat((logits1, logits2 , logits_linear),dim=-1)
			if DEBUG:print("o3 shape" , o3.shape)
			bn3 = nn.BatchNorm1d(768*2+64, affine=False,device=device)
			if self.training:
				o3 = bn3(o3)
			logits3 = self.linear3_p(o3)
			#print(logits1.shape , "WBERT logist1 768 bSE\n")
			#print(logits2.shape , "WBERT logist2 768 bSE\n")
		else:
			o3 =torch.cat((logits1, logits2 ),dim=-1)
			bn3 = nn.BatchNorm1d(768*2, affine=False,device=device)
			o3 = bn3(o3)
			#print("o3 shape 768*2" , o3.shape)
			logits3 = self.linear3(o3)
		#if logits3.shape[0] >1:
		if self.training:
			logits3=self.dropout(logits3)
		if DEBUG:print("logits3 shape 768*2" , logits3.shape)
		logits=self.classifier(logits3)
		#print("logits Calassifier shape 768*2 2 ops" , logits.shape)
		loss = None
		if labels is not None:
			if self.config.problem_type is None:
				if self.num_labels == 1:
					self.config.problem_type = "regression"
				elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
					self.config.problem_type = "single_label_classification"
				else:
					self.config.problem_type = "multi_label_classification"

			if self.config.problem_type == "regression":
				loss_fct = MSELoss()
				if self.num_labels == 1:
					loss = loss_fct(logits.squeeze(), labels.squeeze())
				else:
					loss = loss_fct(logits, labels)
			elif self.config.problem_type == "single_label_classification":
				loss_fct = CrossEntropyLoss()
				loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
			elif self.config.problem_type == "multi_label_classification":
				if DEBUG:print("logits shape" , logits.shape)
				if DEBUG:print("labels shape" , labels.shape) 
				if DEBUG:print("logits " , logits)
				if DEBUG:print("labels" , labels)                 
				loss_fct = BCEWithLogitsLoss()
				#exit()
				loss = loss_fct(logits, labels)
				#print("loss" , loss)
				#print("\n\noutputs" , outs)
				#print("\n\noutputs0[2:]" , outs[0][2:])
				#print("\n\noutputs1[2:]" , outs[1][2:])
		if not return_dict:
			#print("IN WWTF?")
			output = (logits,) + outputs[2:]
			return ((loss,) + output) if loss is not None else output
		
		hs1,hs2 = (torch.Tensor(),torch.Tensor()) if outs[0].hidden_states is None and outs[1].hidden_states is None else (outs[0].hidden_states,outs[1].hidden_states)
		attn1,attn2 = (torch.Tensor(),torch.Tensor()) if outs[0].attentions is None and outs[1].attentions is None else (outs[0].attentions,outs[1].attentions)

		return SequenceClassifierOutput(
			loss=loss,
			logits=logits,
			hidden_states=(hs1,hs2),
			attentions=(attn1,attn2)
		)