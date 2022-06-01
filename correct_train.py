import pandas as pd
import itertools
import json
from bertmodels import Bert,Config
vocab_file = r'/home/xiaoguzai/代码/第五届“达观杯” 基于大规模预训练模型的风险事件标签识别/bert-base-count3-HMCN-F/pretrain/bert_model/vocab.txt'
vocab_size = len(open(vocab_file,'r').readlines()) 
with open('/home/xiaoguzai/数据集/bert-uncased-pytorch/config.json','r',encoding='utf8')as fp:
    json_data = json.load(fp)
#json_data['vocab_size'] = vocab_size
config = Config(**json_data)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from tokenization import FullTokenizer
import numpy as np
vocab_file = "/home/xiaoguzai/代码/第五届“达观杯” 基于大规模预训练模型的风险事件标签识别/bert-base-count3-HMCN-F/pretrain/bert_model/vocab.txt"
tokenizer = FullTokenizer(vocab_file=vocab_file)
from bertmodels import Bert
from tqdm import tqdm
config.with_mlm = False
bertmodel = Bert(config)

import csv
text = []
labels = []
with open('/home/xiaoguzai/数据集/第五届“达观杯” 基于大规模预训练模型的风险事件标签识别数据集/datagrand_2021_train.csv','r') as load_f:
    reader = csv.reader(load_f)
    for row in reader:
        if row[0] != 'id':
            text.append(row[1])
            #标签1～10,1~35
            currentstr = row[2]
            currentstr = currentstr.split('-')
            data1 = int(currentstr[0])
            data2 = int(currentstr[1])
            labels.append((data1-1)*35+(data2-1))

from loader_bert import load_bert_data
bertmodel = load_bert_data(bertmodel,'/home/xiaoguzai/数据集/bert-uncased-pytorch/pytorch_model.bin')
#from loader_pretrain_weights import load_bert_data
#load_bert_data(bertmodel,'/home/xiaoguzai/数据集/第五届“达观杯” 基于大规模预训练模型的风险事件标签识别数据集/filenum=350-model_epoch=0.pth')

class ClassificationDataset(Dataset):
    def __init__(self,text,label,maxlen,flag):
        self.text = text
        self.maxlen = maxlen
        self.label = label
        token_data,token_id,segment_id,mask_id = [],[],[],[]
        #sequence填充可以最后统一实现
        label = []
        for index in tqdm(range(len(self.text))):
        #!!!这里构成ClassificationDataset之中的tqdm还可以优化并行操作
            current_text = text[index]
            current_token = tokenizer.tokenize(current_text)
            current_token = ["[CLS]"]+current_token
            current_id = tokenizer.convert_tokens_to_ids(current_token)
            #current_token = self.sequence_padding(current_token)
            current_id = self.sequence_padding(current_id)
            #current_id = current_id.push(tokenizer.convert_tokens_to_ids(["[SEP]"]))
            current_id = np.append(current_id,tokenizer.convert_tokens_to_ids(["[SEP]"]))
            #current_id = current_id.tolist()
            #!!!!!!!!!!!!!!!这里一定要变成list之后再加上相应的数组，不然就变成加上3了
            token_data.append(current_token)
            #for index1 in range(len(current_id)):
            #    current_id[index1] = (int)(current_id[index1])
            token_id.append(current_id)
            #segment_id.append(current_segment_id)
            #mask_id.append(current_mask_id)
            label.append(self.label[index])
        self.token_data = token_data
        self.token_id = token_id
        #self.segment_id = sequence_padding(self.segment_id,maxlen)
        #self.mask_id = sequence_padding(self.mask_id,maxlen)
        self.tensors = [torch.tensor(self.token_id,dtype=torch.long),
                torch.tensor(self.label,dtype=torch.long)]
        
    def __len__(self):
        return len(self.token_id)
    
    def __getitem__(self,index):
        return tuple(tensor[index] for tensor in self.tensors)
    
    def sequence_padding(self,inputs,padding = 0):
        length = self.maxlen-1
        if len(inputs) > length:
            inputs = inputs[:length]
        outputs = []
        pad_width = (0,length-len(inputs))
        x = np.pad(inputs,pad_width,'constant',constant_values=padding)
        return x

class ClassificationModel(nn.Module):
    def __init__(self,model,config,n_labels):
        super(ClassificationModel,self).__init__()
        #self.embedding = nn.Embedding(30522,768)
        self.model = model
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(config.embedding_size,n_labels)
        
    def forward(self,input_ids,segment_ids,input_mask):
        #outputs = self.embedding(input_ids)
        outputs = self.model(input_ids)
        #[64,128,768]
        outputs = outputs[:,0]
        outputs = self.dropout1(outputs)
        outputs = self.fc2(outputs)
        #outputs = F.softmax(outputs)
        return outputs
    #之前这里少量return outputs返回值为None

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
import numpy as np
ss = StratifiedKFold(n_splits=5,shuffle=True,random_state=2)
#建立4折交叉验证方法 查一下KFold函数的参数
text = np.array(text)
labels = np.array(labels)
for train_index,test_index in ss.split(text,labels):
    train_text = text[np.array(train_index)]
    test_text = text[test_index]
    train_labels = labels[train_index]
    test_labels = labels[test_index]
train_dataset = ClassificationDataset(train_text,train_labels,maxlen=32,flag=True)
test_dataset = ClassificationDataset(test_text,test_labels,maxlen=32,flag=False)
#到里面的classificationdataset才进行字符的切割以及划分
train_loader = DataLoader(train_dataset,batch_size=64,shuffle=True)
test_loader = DataLoader(test_dataset,batch_size=64)

from loader_bert import load_bert_data
from tqdm import tqdm
from colorama import Fore
model = ClassificationModel(bertmodel,config,350)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_func = torch.nn.CrossEntropyLoss()

final_data_label = []
final_true = []
final_predict = []
final_reality = []
for data in range(350):
    final_data_label.append(data)

import torch.nn.functional as F
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(3):
    print('epoch {}'.format(epoch+1))
    train_loss = 0
    train_acc = 0
    
    model.train()
    
    model = model.to(device)
    model = nn.DataParallel(model)
    for batch_token_ids,batch_labels in tqdm(train_loader):
        batch_token_ids = batch_token_ids.to(device)
        batch_labels = batch_labels.to(device)
        output = model(batch_token_ids,None,None)
        optimizer.zero_grad()
        #loss = loss_func(output,batch_labels)
        loss = loss_func(output,batch_labels)
        train_loss = train_loss+loss
        loss.backward()
        optimizer.step()
    print('Train Loss: {:.6f}, Acc: {:.6f}'.format(train_loss, train_acc/len(train_dataset)))

    model.eval()
    eval_loss = 0.
    eval_acc = 0.
    #!!!
    eval_predict_label = []
    for batch_token_ids,batch_labels in test_loader:
        batch_token_ids = batch_token_ids.to(device)
        #batch_segment_ids = batch_segment_ids.to(device)
        #batch_mask_ids = batch_mask_ids.to(device)
        batch_labels = batch_labels.to(device)
        with torch.no_grad():
            #output = model(batch_token_ids,batch_segment_ids,batch_mask_ids)
            output = model(batch_token_ids,None,None)
        loss = loss_func(output, batch_labels)
        eval_loss += loss
        pred = torch.max(output, 1)[1]
        print('pred = ')
        print(pred)
        #!!!
        eval_predict_label.extend(pred.cpu())
        num_correct = (pred == batch_labels).sum()
        eval_acc += num_correct
    torch.cuda.empty_cache() 
    print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss, eval_acc/len(test_dataset)))