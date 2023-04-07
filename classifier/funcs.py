import re
from sklearn.metrics import accuracy_score, f1_score , confusion_matrix , precision_recall_fscore_support , precision_score , recall_score , precision_recall_curve
import numpy as np
import torch
def compute_metrics_tests(pred):
		is_regression=False    
		#print("Compute Metrics\n")
		logits, labels = pred
		#print("logits1" ,logits )
		#print("labels1" , labels)
		logits = logits[0] if isinstance(logits , tuple) else logits
		#preds2 = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
		##print("preds2" , preds2)
		#preds2 = np.squeeze(preds2) if is_regression else np.argmax(preds2, axis=1)
		##print("preds2" , preds2)
		#print("logits" ,logits)
		logits= logits.detach().argmax(axis=1).cpu()
		print("CHECK logits" , logits , logits.shape)
		##print("CHECK2 ", np.argmax(pred.predictions[0] , axis = -1))

		##print("\n#print(pred.predictions\n",pred.predictions)
		##print("\n#print(pred.label_ids\n",pred.label_ids)
		##print(pred.predictions.shape)
		##print(pred.label_ids.shape)
	
		#labels = pred.label_ids
		#preds = pred.predictions.argmax(-1)
		labels= torch.tensor(labels.to_numpy()).detach().cpu()#.argmax(axis=-1)
		print("labels" , labels , labels.shape)
		
		f1 = f1_score(labels, logits, average="weighted")
		acc = accuracy_score(labels, logits)
		_precision_score = precision_score(labels, logits)
		_recall_score = recall_score(labels, logits)
		_confusion = confusion_matrix(labels, logits)
		#print({"mse": ((logits.float() - labels.astype(float)) ** 2).mean().item()})
		#print({"accuracy": acc, "f1": f1 ,"precision" :_precision_score , "recall":_recall_score , 'CM':_confusion })
		"""#print({"accuracy": type(acc), "f1": type(f1) ,"precision" :type(_precision_score) , "recall":type(_recall_score) 
		, 'CM':type(_confusion) ,"mse": ((logits - labels) ** 2).mean().item()}})"""
		#print(dir(acc))
		#print()
		#print(dir(_confusion))
		"""
		acc=acc.item()
		f1=f1.item()
		_precision_score=_precision_score.item()
		_recall_score=_recall_score.item()"""
		_confusion=_confusion.tolist()
		return {"accuracy": acc, "f1": f1 ,"precision" :_precision_score , "recall":_recall_score , 'CM':_confusion }

def compute_metrics(pred):
		is_regression=False    
		#print("Compute Metrics\n")
		logits, labels = pred
		#print("logits1" ,logits )
		#print("labels1" , labels)
		logits = logits[0] if isinstance(logits , tuple) else logits
		#preds2 = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
		##print("preds2" , preds2)
		#preds2 = np.squeeze(preds2) if is_regression else np.argmax(preds2, axis=1)
		##print("preds2" , preds2)
		#print("logits" ,logits)
		logits= logits.argmax(axis=-1)
		#print("CHECK logits" , logits)
		#print("CHECK2 ", np.argmax(pred.predictions[0] , axis = -1))

		##print("\n#print(pred.predictions\n",pred.predictions)
		##print("\n#print(pred.label_ids\n",pred.label_ids)
		##print(pred.predictions.shape)
		##print(pred.label_ids.shape)
	
		#labels = pred.label_ids
		#preds = pred.predictions.argmax(-1)
		labels= labels.argmax(axis=-1)
		#print("labels" , labels)
		
		f1 = f1_score(labels, logits, average="weighted")
		acc = accuracy_score(labels, logits)
		_precision_score = precision_score(labels, logits)
		_recall_score = recall_score(labels, logits)
		_confusion = confusion_matrix(labels, logits)
		#print({"mse": ((logits - labels) ** 2).mean().item()})
		#print({"accuracy": acc, "f1": f1 ,"precision" :_precision_score , "recall":_recall_score , 'CM':_confusion })
		"""#print({"accuracy": type(acc), "f1": type(f1) ,"precision" :type(_precision_score) , "recall":type(_recall_score) 
		, 'CM':type(_confusion) ,"mse": ((logits - labels) ** 2).mean().item()}})"""
		#print(dir(acc))
		#print()
		#print(dir(_confusion))
		"""
		acc=acc.item()
		f1=f1.item()
		_precision_score=_precision_score.item()
		_recall_score=_recall_score.item()"""
		_confusion=_confusion.tolist()
		return {"accuracy": acc, "f1": f1 ,"precision" :_precision_score , "recall":_recall_score , 'CM':_confusion }
def message_length(x):
	# returns total number of characters
	return len(x.strip())

def num_capitals(x):
	_, count = re.subn(r'[A-Z]', '', x.strip()) # only works in english
	return count

def num_punctuation(x):
	_, count = re.subn(r'\W', '', x.strip())
	return count
def word_count(x):    
	b=re.split(r'[^a-zA-Z0-9]',x.strip())#b=re.split(r'[^a-zA-Z]',x)
	return len(list(filter(None,b)))