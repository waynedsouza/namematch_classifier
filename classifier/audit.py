import csv
import pandas as pd 
import sys
import torch 
import numpy as np
from funcs import compute_metrics_tests
from transformers import BertForSequenceClassification

from transformers import AutoModel
sys.path.insert(0, 'F:\\Pycode\\canadarealestate\\')##needs moule canadarealestate
sys.path.insert(0, 'F:\\Pycode\\')
from util_conv import Conv_wayne
st_model="F:\\Pycode\\canadarealestate\\wbertcheckpoint.tar\\wbertcheckpoint\\blah6-blah6-indep\\output\\blah6-blah6-indep-best_acc_--2023-02-25"
st_model2="F:\\Pycode\\canadarealestate\\wbertcheckpoint.tar\\wbertcheckpoint\\blah6-blah6-indep\\output\\blah6-blah6-indep-best_loss_--2023-03-30"
st_model3="F:\\Pycode\\canadarealestate\\wbertcheckpoint.tar\\wbertcheckpoint\\blah6-blah6-indep\\output\\blah6-blah6-indep-best_scores_--2023-03-30"
cpt_model = "F:\\Pycode\\canadarealestate\\wbertcheckpoint.tar\\wbertcheckpoint\\blah6-blah6-indep\\checkpoint-7600"
cpt_model = "F:\\Pycode\\canadarealestate\\model_pre.tar\\model_pre"
#ZMODEL3=torch.load(st_model3 ,"cpu")
#ZMODEL2=torch.load(st_model2 ,"cpu")
model = BertForSequenceClassification.from_pretrained(cpt_model,local_files_only=True,ignore_mismatched_sizes=True)
#ZMODEL=torch.load(st_model ,"cpu")
#
model.eval()

#ZMODEL3.eval()#, 
#ZMODEL2.eval()
#ZMODEL.eval()
#model.eval()
convetor = Conv_wayne()
self=convetor
#.prepare()
data="F:\\Pycode\\canadarealestate\\data\\"
gold = "GOLD_POSITIVES_phone_list.csv"
path = data+gold

"""
rows=[]
with open(path , 'r') as f:    
    for row in csv.reader(f):
        rows.append(row)"""
df = pd.read_csv(path)
df['Capitals1'] = df['name'].apply(self.num_capitals)
df['Punctuation1'] = df['name'].apply(self.num_punctuation)
df['Length1'] = df['name'].apply(self.message_length)
df['WordCount1'] = df['name'].apply(self.word_count)
# df['Words1'] = df['sentence1'].apply(word_counts)
#df['Words1V3'] = df['sentence1'].apply(word_counts_v3)

df['Capitals2'] = df['nameofperson'].apply(self.num_capitals)
df['Punctuation2'] = df['nameofperson'].apply(self.num_punctuation)
df['Length2'] = df['nameofperson'].apply(self.message_length)
#df['Words2'] = df['sentence2'].apply(word_counts)
#df['Words2V3'] = df['sentence2'].apply(word_counts_v3)
df['WordCount2'] = df['nameofperson'].apply(self.word_count)
x=df.loc[: , ['name', 'Capitals1' ,'Punctuation1' , 'Length1' , 'WordCount1' ]]
y=df.loc[: , ['nameofperson', 'Capitals2' ,'Punctuation2' , 'Length2' , 'WordCount2' ]]
self.prepare()
features = x,y 
features2 = y,x
ret1 = self.wtokenizer.prepare(features)
ret2 = self.wtokenizer.prepare(features2)
LABELS = np.stack([np.zeros(159 ) , np.ones(159)] , axis = 1)

"""zo3=ZMODEL3(**ret1) 
zo3.logits = zo3.logits.double()
op3 = zo3.logits.argmax(axis=1)
LABELS = np.stack([np.zeros(159 ) , np.ones(159)] , axis = 1)
c3 = compute_metrics_tests((zo3.logits , LABELS))"""


#zo3_2=ZMODEL3(**ret2)
#zo3_2.logits = zo3_2.logits.double()
#op3 = zo3_2.logits.argmax(axis=1)
#
#c3_2 = compute_metrics_tests((zo3_2.logits , LABELS))
"""
zo2=ZMODEL2(**ret1)
zo2.logits = zo2.logits.double()
c2 = compute_metrics_tests((zo2.logits , LABELS))


 
zo2_2=ZMODEL2(**ret2) 
zo2_2.logits = zo2_2.logits.double()
c2_2 = compute_metrics_tests((zo2_2.logits , LABELS))"""

"""
zo=ZMODEL(**ret1) 
zo.logits = zo.logits.double()
c1 = compute_metrics_tests((zo.logits , LABELS))"""



#op3_2 = zo3_2.logits.argmax(axis=1)


#c3_2 = compute_metrics_tests((zo3_2.logits , LABELS))
"""
zo_2=ZMODEL(**ret2)
zo_2.logits = zo_2.logits.double()
c2_2 = compute_metrics_tests((zo_2.logits , LABELS))"""


zo=model(**ret1) 
zo.logits = zo.logits.double()
c1 = compute_metrics_tests((zo.logits , LABELS))

zo_2=model(**ret2) 
zo_2.logits = zo_2.logits.double()
c1 = compute_metrics_tests((zo_2.logits , LABELS))