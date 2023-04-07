from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
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
			_ip1 , _ip2= input_dict[self.main_input_name]
			print("_ip1 , _ip2",_ip1 , _ip2)
			print("_ip1 , _ip2" , _ip1.shape , _ip2.shape)
			print("_ip1 , _ip2 numel" , _ip1.numel() , _ip2.numel())
			print("_ip1 , _ip2 numelplus " , _ip1.numel() + _ip2.numel())
			return _ip1.numel() + _ip2.numel()
			#return input_dict[self.main_input_name].numel()
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
		print("input_dict",input_dict)

		return 6 * self.estimate_tokens(input_dict) * self.num_parameters(exclude_embeddings=exclude_embeddings)