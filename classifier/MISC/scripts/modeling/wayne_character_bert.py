# Functions are adapted from Huggingface's transformers library:
# https://github.com/allenai/allennlp/

""" Defines the main CharacterBERT PyTorch class. """
import torch
from torch import nn
import sys
#from transformers.modeling_bert import BertPreTrainedModel, BertEncoder, BertPooler
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertEncoder, BertPooler
from transformers.modeling_outputs  import BaseModelOutputWithPoolingAndCrossAttentions
#sys.path.insert(0, 'F:\\Pycode\\canadarealestate\\characterbert\\character_bert')
#sys.path.insert(0 ,"F:\\Pycode\\canadarealestate\\characterbert\\pretrained-models\\medical_character_bert")
from .character_cnn import CharacterCNN



class BertCharacterEmbeddings(nn.Module):
	""" Construct the embeddings from char-cnn, position and token_type embeddings. """
	def __init__(self, config):
		super(BertCharacterEmbeddings, self).__init__()

		# This is the module that computes word embeddings from a token's characters
		self.word_embeddings = CharacterCNN(
			requires_grad=True,
			output_dim=config.hidden_size)

		self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
		self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

		# self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
		# any TensorFlow checkpoint file
		self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
		self.dropout = nn.Dropout(config.hidden_dropout_prob)

	def forward(self, input_ids, token_type_ids=None, position_ids=None,inputs_embeds = None, past_key_values_length= 0,):
		seq_length = input_ids[:, :, 0].size(1)
		#print(seq_length , "bertcharembed  seq_length int eg.7\n")
		if position_ids is None:
			position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids[:, :, 0].device)
			#print(position_ids.shape , "bertcharembed position_ids no batches\n")
			position_ids = position_ids.unsqueeze(0).expand_as(input_ids[:, :, 0])
		if token_type_ids is None:
			token_type_ids = torch.zeros_like(input_ids[:, :, 0])
			#print(token_type_ids.shape , "bertcharembed token_type_ids BSE\n")

		if inputs_embeds is None:
			inputs_embeds = self.word_embeddings(input_ids)

		words_embeddings = self.word_embeddings(input_ids)
		#print(words_embeddings.shape , "bertcharembed words_embeddings BSE\n")
		position_embeddings = self.position_embeddings(position_ids)
		#print(position_embeddings.shape , "bertcharembed position_embeddings BSE\n")
		token_type_embeddings = self.token_type_embeddings(token_type_ids)
		#print(token_type_embeddings.shape , "bertcharembed token_type_embeddings BSE\n")

		embeddings = words_embeddings + position_embeddings + token_type_embeddings
		#print(embeddings.shape , "bertcharembed embeddings BSE\n")
		embeddings = self.LayerNorm(embeddings)
		embeddings = self.dropout(embeddings)
		#print(embeddings.shape , "bertcharembed final_embeddings BSE\n")
		return embeddings


class CharacterBertModel(BertPreTrainedModel):
	""" BertModel using char-cnn embeddings instead of wordpiece embeddings. """

	def __init__(self, config, add_pooling_layer=True):
		super().__init__(config)
		self.config = config

		self.embeddings = BertCharacterEmbeddings(config)#for change
		self.encoder = BertEncoder(config)#check
		self.pooler = BertPooler(config)#check

		self.init_weights()
		self.post_init()#check where this comes in latest transformers it should init the BertCharacterEmbeddings layer

	def get_input_embeddings(self):
		return self.embeddings.word_embeddings

	def set_input_embeddings(self, value):
		self.embeddings.word_embeddings = value

	def _prune_heads(self, heads_to_prune):
		for layer, heads in heads_to_prune.items():
			self.encoder.layer[layer].attention.prune_heads(heads)
			"""
	@add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
	@add_code_sample_docstrings(
		checkpoint=_CHECKPOINT_FOR_DOC,
		output_type=BaseModelOutputWithPoolingAndCrossAttentions,
		config_class=_CONFIG_FOR_DOC,
	)"""

	def forward(
		self,
		input_ids=None,
		attention_mask=None,
		token_type_ids=None,
		position_ids=None,
		head_mask=None,
		inputs_embeds=None,
		encoder_hidden_states=None,
		encoder_attention_mask=None,
		past_key_values = None,
		use_cache= None,
		output_attentions= None,
		output_hidden_states= None,
		return_dict = None,
		**kwargs
	):

		output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
		output_hidden_states = (
			output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
		)
		return_dict = return_dict if return_dict is not None else self.config.use_return_dict
		if self.config.is_decoder:
			use_cache = use_cache if use_cache is not None else self.config.use_cache
		else:
			use_cache = False

		if input_ids is not None and inputs_embeds is not None:
			raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
		elif input_ids is not None:
			input_shape = input_ids[:,:,0].size()
		elif inputs_embeds is not None:
			input_shape = inputs_embeds.size()[:-1]
		else:
			raise ValueError("You have to specify either input_ids or inputs_embeds")

		batch_size, seq_length = input_shape
		device = input_ids.device if input_ids is not None else inputs_embeds.device
		# past_key_values_length
		past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

		if attention_mask is None:
			attention_mask = torch.ones(input_shape, device=device)
		if token_type_ids is None:
			if hasattr(self.embeddings, "token_type_ids"):
				raise Exception("Ist condition")
				buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
				buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
				token_type_ids = buffered_token_type_ids_expanded
			else:
				token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
				#token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

		# We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
		# ourselves in which case we just need to make it broadcastable to all heads.
		if attention_mask.dim() == 3:
			extended_attention_mask = attention_mask[:, None, :, :]
		elif attention_mask.dim() == 2:
			# Provided a padding mask of dimensions [batch_size, seq_length]
			# - if the model is a decoder, apply a causal mask in addition to the padding mask
			# - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
			if self.config.is_decoder:
				batch_size, seq_length = input_shape
				seq_ids = torch.arange(seq_length, device=device)
				causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
				causal_mask = causal_mask.to(
					attention_mask.dtype
				)  # causal and attention masks must have same type with pytorch version < 1.3
				extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
			else:
				extended_attention_mask = attention_mask[:, None, None, :]
		else:
			raise ValueError(
				"Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(
					input_shape, attention_mask.shape
				)
			)

		# Since attention_mask is 1.0 for positions we want to attend and 0.0 for
		# masked positions, this operation will create a tensor which is 0.0 for
		# positions we want to attend and -10000.0 for masked positions.
		# Since we are adding it to the raw scores before the softmax, this is
		# effectively the same as removing these entirely.
		extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
		extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

		# If a 2D ou 3D attention mask is provided for the cross-attention
		# we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
		if self.config.is_decoder and encoder_hidden_states is not None:
			encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
			encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
			if encoder_attention_mask is None:
				encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)

			if encoder_attention_mask.dim() == 3:
				encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
			elif encoder_attention_mask.dim() == 2:
				encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
			else:
				raise ValueError(
					"Wrong shape for encoder_hidden_shape (shape {}) or encoder_attention_mask (shape {})".format(
						encoder_hidden_shape, encoder_attention_mask.shape
					)
				)

			encoder_extended_attention_mask = encoder_extended_attention_mask.to(
				dtype=next(self.parameters()).dtype
			)  # fp16 compatibility
			encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -10000.0
		else:
			encoder_extended_attention_mask = None

		# Prepare head mask if needed
		# 1.0 in head_mask indicate we keep the head
		# attention_probs has shape bsz x n_heads x N x N
		# input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
		# and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
		if head_mask is not None:
			if head_mask.dim() == 1:
				head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
				head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
			elif head_mask.dim() == 2:
				head_mask = (
					head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
				)  # We can specify head_mask for each layer
			head_mask = head_mask.to(
				dtype=next(self.parameters()).dtype
			)  # switch to fload if need + fp16 compatibility
		else:
			head_mask = [None] * self.config.num_hidden_layers
		"""
		embedding_output = self.embeddings(
			input_ids=input_ids, 
			position_ids=position_ids,
			token_type_ids=token_type_ids
		)"""
		#print("inputs_embeds_-",inputs_embeds)
		embedding_output = self.embeddings(
			input_ids=input_ids,
			position_ids=position_ids,
			token_type_ids=token_type_ids,
			inputs_embeds=inputs_embeds,
			past_key_values_length=past_key_values_length,
		)
		encoder_outputs = self.encoder(
			embedding_output,
			attention_mask=extended_attention_mask,
			head_mask=head_mask,
			encoder_hidden_states=encoder_hidden_states,
			encoder_attention_mask=encoder_extended_attention_mask,
			past_key_values=past_key_values,
			use_cache=use_cache,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			return_dict=return_dict,
		)
		"""
		encoder_outputs = self.encoder(
			embedding_output,
			attention_mask=extended_attention_mask,
			head_mask=head_mask,
			encoder_hidden_states=encoder_hidden_states,
			encoder_attention_mask=encoder_extended_attention_mask,
		)"""
		sequence_output = encoder_outputs[0]
		#pooled_output = self.pooler(sequence_output)
		pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

		if not return_dict:
			return (sequence_output, pooled_output) + encoder_outputs[1:]

		"""outputs = (sequence_output, pooled_output,) + encoder_outputs[
			1:
		]"""  # add hidden_states and attentions if they are here
		return BaseModelOutputWithPoolingAndCrossAttentions(
			last_hidden_state=sequence_output,
			pooler_output=pooled_output,
			past_key_values=encoder_outputs.past_key_values,
			hidden_states=encoder_outputs.hidden_states,
			attentions=encoder_outputs.attentions,
			cross_attentions=encoder_outputs.cross_attentions,
		)
		#return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)


if __name__ == "__main__":
	import logging
	from download import download_model
	logging.basicConfig(level=logging.INFO)

	#download_model('medical_character_bert')
	#path = "pretrained-models/medical_character_bert/"
	path="F:\\Pycode\\canadarealestate\\characterbert\\pretrained-models\\medical_character_bert"

	model = CharacterBertModel.from_pretrained(path)
	print(model ,"__MODEL")
	logging.info('%s', model)
