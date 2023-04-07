import torch 
from torch import nn
import sys
sys.path.insert(0, '../characterbert/character_bert/')
from utils.character_cnn import CharacterIndexer
from transformers import BertTokenizer

class WayneCharacterModel(nn.Module):

    def __init__(self, model_path='pretrained-models/general_character_bert/' , dropout=0.1):
        super(WayneCharacterModel, self).__init__()
        self.character_bert_model = CharacterBertModel.from_pretrained(model_path)
        """
        if type(model_path) is SentenceTransformer:
          self.model = model_path
        else:
          self.model = SentenceTransformer(model_path)
        _modules = next(self.model.modules())
        _final = _modules[-1]
        if type(_final) is models.Pooling:
          _num_features = _final.pooling_output_dimension
        else:
          _num_features = _final.out_features
        print(_num_features ," num features ip WayneClassifier")
        """
        _num_features=768
        self.indexer = CharacterIndexer()
        self.dropout = nn.Dropout(dropout)
        #self.dropout2 = nn.Dropout(dropout)
        #self.dropout3 = nn.Dropout(dropout)
        self.linear1 = nn.Linear(_num_features*2,_num_features )
        #self.linear2 = nn.Linear(768,256 )
        #self.linear3 = nn.Linear(256,1 )
        nn.init.xavier_normal_(self.linear1.weight)
        #nn.init.xavier_normal_(self.linear2.weight)
        #nn.init.xavier_normal_(self.linear3.weight)
        #self.relu = nn.ReLU()
        #self.relu2 = nn.ReLU()
        self.tanh = nn.Tanh()
        #self.sigmoid1 = nn.Sigmoid()
        #self.sigmoid2 = nn.Sigmoid()
        #self.sigmoid3 = nn.Sigmoid()
        #self.smart_batching_collate=model_path.smart_batching_collate
        self.use_cuda = torch.cuda.is_available()
        self.device= torch.device("cuda" if self.use_cuda else "cpu")
        self.bert_tokenizer= BertTokenizer.from_pretrained('bert-base-uncased')
        self.indexer = CharacterIndexer()
        print("Inited WayneCharacterModel")
        

    def forward(self, input_id, mask=None):
        print("Forward WayneCharacterModel ",input_id)
        ip1,ip2 = {} ,{}
        if mask is not None and type(input_id) is tuple and len(input_id) == 2 and type(input_id[0]) is torch.Tensor:
          raise Exception("Not Coded")
          (input_id1 ,input_id2) , (mask1 , mask2) = input_id, mask
          print("\ninput_id1" , input_id1 , " \n shape " , input_id1.shape)
          
          ip1['input_ids']=input_id1
          ip1['attention_mask'] = mask1

          ip2['input_ids']=input_id2
          ip2['attention_mask'] = mask2
        else:
          assert(mask is None)
          assert(len(input_id) == 2)
          lhs , rhs = input_id
          t_lhs = [self.bert_tokenizer.basic_tokenizer.tokenize(i) for i in lhs]
          t_rhs = [self.bert_tokenizer.basic_tokenizer.tokenize(i) for i in rhs]
          t_lhs = [['[CLS]', *i, '[SEP]'] for i in t_lhs]
          t_rhs = [['[CLS]', *i, '[SEP]'] for i in t_rhs]
          print("\n\n")
          print(t_lhs,"\nT_lhs rhs ")
          input_tensor_lhs = self.indexer.as_padded_tensor(t_lhs).to(self.device)
          input_tensor_rhs = self.indexer.as_padded_tensor(t_rhs).to(self.device)
          print("\n\n")
          print(input_tensor_lhs,"\input_tensor_lhs rhs ")
          print(len(input_tensor_lhs) ,"__LEN Input Tensor")
          lhs = self.character_bert_model(input_tensor_lhs)
          rhs = self.character_bert_model(input_tensor_rhs)
          out_tensor =torch.cat((lhs[1],rhs[1]),dim=-1)
          print("\n\n")
          print(out_tensor,"\out_tensor ")
          print("\n\n")
          print(out_tensor.shape , "Cats shape")
          linear_output1 = self.linear1(out_tensor)
          acts1 = self.tanh(linear_output1)
          dropout1 = self.dropout(acts1)
          print(dropout1.shape , "Rroupout shape")
          """
          if type(lhs) is str and type(rhs) is str:
            lhs = self.model.tokenize([lhs])
            rhs = self.model.tokenize([rhs])

            ip1['input_ids']=lhs['input_ids']
            ip1['attention_mask'] = lhs['attention_mask']

            ip2['input_ids']=rhs['input_ids']
            ip2['attention_mask']= rhs['attention_mask']
            
          else:
            lhs = self.model.tokenize(lhs)
            rhs = self.model.tokenize(rhs)
            ip1['input_ids']=lhs['input_ids']
            ip1['attention_mask'] = lhs['attention_mask']

            ip2['input_ids']=rhs['input_ids']
            ip2['attention_mask']= rhs['attention_mask']
          print(ip1 , type(ip1),"__HERE")
          
          device = torch.device("cuda" if use_cuda else "cpu")
          ip1['input_ids'] = ip1['input_ids'].to(device)
          ip2['input_ids'] = ip2['input_ids'].to(device)
          ip1['attention_mask']=ip1['attention_mask'].to(device)
          ip2['attention_mask']=ip2['attention_mask'].to(device)
        
        out1 = self.model(ip1)
        out2 = self.model(ip2)
        print("out1: " ,out1['sentence_embedding'].shape , "\n sentence_embedding shape")
        print("out2: ", out2['sentence_embedding'].shape , "\n sentence_embedding shape")
        out=torch.cat((out1['sentence_embedding'] ,out2['sentence_embedding']) ,dim=1)
        print("out: " , out , "\n shapeOUT", out.shape)
         #_, pooled_output1
        
        
        dropout_output1 = self.dropout1(out)
        print("Linear ip 1 shape" , dropout_output1.shape)
        
        acts1 = self.relu(linear_output1)
        dropout_output2 = self.dropout2(acts1)
        linear_output2 = self.linear2(dropout_output2)
        acts2 = self.relu2(linear_output2)
        dropout_output3 = self.dropout3(acts2)
        
        linear_output3 = self.linear3(dropout_output3)
        final_layer = self.sigmoid3(linear_output3)
        """

        return dropout1