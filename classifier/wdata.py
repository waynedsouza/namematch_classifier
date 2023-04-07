import os
import pandas as pd
from torch.utils.data import Dataset
import torch	
#from torchvision.io import read_image


class Wdata(Dataset):
	def __init__(self , sentences=None , sentences2=None , labels=None , multicalss_labels=True):
		super(Wdata , self).__init__()
		self.sentences=None
		self.labels=None
		self.keys=None
		self.lhs , self.rhs = None,None
		self.multicalss_labels = multicalss_labels
		if sentences is not None and sentences2 is not None:
			self.sentences =list(zip(sentences , sentences2)) 
		if labels is not None:
			self.labels = labels
			if self.multicalss_labels is True:
				self.labels= list(map(self.MClabel , self.labels))
		if self.sentences is not None and self.labels is not None:
			self.frame = pd.DataFrame({'label_ids' : self.sentences , 'label':self.labels })
		self.pin_memory = False if torch.cuda.is_available() else True

	def MClabel(self,_label):
		if _label ==1:
			return [0.,1.]
		return [1.,0.]
	def from_df(self , df ,label= 'label', ignore_columns =[]):		
		all_columns= df.columns.to_list()
		self.all_keys=[i for i in all_columns if i !=  label and type(i) is str and (i[-1:]).isdecimal() and ((i[-1:])=='1' or (i[-1:])=='2')]
		canon_key = list(set([i[:-1] for i in self.all_keys]))
		canon_key.sort()
		canon_key= [i for i in canon_key if i+'1' in self.all_keys and i+'2' in self.all_keys]
		if 'sentence' in canon_key:
			canon_key=['sentence']+[i for i in canon_key if i !='sentence']
		keys1=[i+'1' for i in canon_key]
		keys2=[i+'2' for i in canon_key]
		if label in all_columns:
			self.labels=df[label].to_list()
		if self.multicalss_labels is True:
			self.labels= list(map(self.MClabel , self.labels))
			#self.labels = [self.MClabel(i) for i in self.labels]
		self.lhs= df[keys1]
		self.rhs= df[keys2]


	def __len__(self):
		return len(self.labels)
	def __getitem__(self, idx):
		if self.lhs is not None and self.rhs is not None:
			features = tuple(self.lhs.iloc[idx].to_list()) , tuple(self.rhs.iloc[idx].to_list())
			label = self.labels[idx]
			return features, label
		features = self.frame.iloc[idx , 0]
		label = self.frame.iloc[idx , 1]        
		return features, label