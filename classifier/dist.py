import csv
import pandas as pd
import glob
import os
import csv
from sentence_transformers import SentenceTransformer , util

from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer , util

#import MySQLdb as mysql
#from dbops import DbOps
file_path=os.path.join("F:\\Pycode\\canadarealestate\\data\\", "*.csv")

#file_path="article_lists/*.csv"
def fuckCSV(x):
    try:
        return pd.read_csv(x,encoding='unicode_escape',delimiter=",",lineterminator="\n")
    except Exception as e:
        print(e)
#file_path='../data/'
#a =listdir(file_path)
types=['GOLD_POSITIVES','NEGATIVES' , 'SILVER_POSITIVES']
files = glob.glob(file_path)
b = map(fuckCSV, files)
c=list(b)
#[i[0]['person']=i[0]['name']+" "+i[0]['postal_address'] for i in c]
for idx,i in enumerate(c):
    c[idx]['x1_person_search']=i['name']+" "+i['address']+" " + i["pincode"]
    c[idx]['x1_rev_person_search']= i['address']+" " + i["pincode"]+" " + i['name']

    c[idx]['x0_person_canon']=i['nameofperson']+" "+i['postal_address']
    c[idx]['x0_rev_person_canon']= i['postal_address']+" "+ i['nameofperson']



    c[idx]['x1_person_search_mangled']=i['name']+i['address']+ i["pincode"]
    c[idx]['x1_rev_person_search_mangled']= i['address']+ i["pincode"] + i['name']

    c[idx]['x0_person_canon_mangled']=i['nameofperson']+i['postal_address']
    c[idx]['x0_rev_person_canon_mangled']= i['postal_address']+ i['nameofperson']
    c[idx]['y']=0 if idx==1 else 1

a = c[0].loc[: , ['x1_person_search','x0_person_canon']]
b = c[2].loc[: , ['x1_person_search','x0_person_canon']]
neg = c[1].loc[: , ['x1_person_search','x0_person_canon']]
sentences=[(a['x0_person_canon'][i] , a['x1_person_search'][i] ) for i in a.index]
silver_sentences=[(b['x0_person_canon'][i] , b['x1_person_search'][i] ) for i in b.index]
neg_sentences=[(neg['x0_person_canon'][i] , neg['x1_person_search'][i] ) for i in neg.index]
"""
for i in a.index:
    print(i , a['x1_person_search'][i])"""
"""
for i in range(len(a)):
    print(a.loc[i, "x1_person_search"], df.loc[i, "x0_person_canon"])
    print(df.iloc[i, 0], df.iloc[i, 2])

for index, row in df.iterrows():
    print(row["x1_person_search"], row["x0_person_canon"])"""

model = SentenceTransformer('sentence-transformers/all-roberta-large-v1')
embeddings = [model.encode(i) for i in sentences]
silver_embeddings = [model.encode(i) for i in silver_sentences]
neg_embeddings =[model.encode(i) for i in neg_sentences] 

scores = [util.dot_score(*i) for i in embeddings]
scores = [i.item() for i in scores]# 0.49750828742980957 => 0.9074708819389343

b_scores = [util.dot_score(*i) for i in silver_embeddings]
b_scores = [i.item() for i in b_scores]# 0.49750828742980957 => 0.9074708819389343

neg_scores = [util.dot_score(*i) for i in neg_embeddings]
neg_scores = [i.item() for i in neg_scores]# 0.49750828742980957 => 0.9074708819389343

print("Gold max:%s - min:%s len:%s"%(max(scores) ,min(scores) , len(scores)))
print("Silver max:%s - min:%s len:%s"%(max(b_scores) ,min(b_scores) , len(b_scores)))
print("Negs max:%s - min:%s len:%s"%(max(neg_scores) ,min(neg_scores) , len(neg_scores) ))


for sentence, embedding in zip(sentences, embeddings):
    print("Sentence:", sentence)
    print("Embedding:", embedding)
    print("")




#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# Sentences we want sentence embeddings for
#sentences = ['This is an example sentence', 'Each sentence is converted']

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-roberta-large-v1')
model = AutoModel.from_pretrained('sentence-transformers/all-roberta-large-v1')
#encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
encoded_input_list = [tokenizer(i, padding=True, truncation=True, return_tensors='pt') for i in sentences]
       
# Compute token embeddings
with torch.no_grad():
    model_output = model(**encoded_input)