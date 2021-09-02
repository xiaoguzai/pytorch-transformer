import pandas as pd
import itertools
import csv
text1,text2,labels = [],[],[]
with open('/home/xiaoguzai/数据集/glue/glue_data/MRPC/train.tsv', 'r') as f:
    lines = csv.reader(f, delimiter='\t', quotechar=None)
    for line in itertools.islice(lines,1,None):
        #第0行是名称，直接切掉
        labels.append(int(line[0]))
        text1.append(line[3])
        text2.append(line[4])

import json
from bertmodels import Bert,Config
with open('/home/xiaoguzai/数据集/bert-uncased-pytorch/config.json','r',encoding='utf8')as fp:
    json_data = json.load(fp)
print(json_data)
config = Config(**json_data)
r"""
test(**kwargs)** 的作用则是把字典 kwargs 变成关键字参数传递。比如上面这个代码，
如果 kwargs 等于 {'a':1,'b':2,'c':3} ，那这个代码就等价于 test(a=1,b=2,c=3) 
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from tokenization import FullTokenizer
import numpy as np
bert_ckpt_dir="/home/xiaoguzai/数据集/chinese-bert/"
bert_ckpt_file = bert_ckpt_dir + "bert_model.ckpt"
bert_config_file = bert_ckpt_dir + "bert_config.json"
vocab_file = bert_ckpt_dir + "vocab.txt"
tokenizer = FullTokenizer(vocab_file=vocab_file)
class ClassificationDataset(Dataset):
    def __init__(self,text1,text2,labels,maxlen):
        self.text1 = text1
        self.text2 = text2
        self.labels = labels
        self.maxlen = maxlen
        token_id,segment_id,mask_id = [],[],[]
        for index in range(len(self.text1)):
            text1 = self.text1[index]
            text2 = self.text2[index]
            token1 = tokenizer.tokenize(text1)
            token1 = ["[CLS]"]+token1+["[SEP]"]
            token_id1 = tokenizer.convert_tokens_to_ids(token1)
            token2 = tokenizer.tokenize(text2)
            token2 = token2+["[SEP]"]
            token_id2 = tokenizer.convert_tokens_to_ids(token2)
            current_token_id = token_id1+token_id2
            current_mask_id = [1]*len(current_token_id)
            #print('before sequence_padding')
            #print('current_token_id = ')
            #print(current_token_id)
            current_token_id = self.sequence_padding(current_token_id)
            #print('after sequence_padding')
            #print('current_token_id = ')
            #print(current_token_id)
            token_id.append(current_token_id)
            segment_id1 = [0]*len(token_id1)
            segment_id2 = [1]*len(token_id2)
            current_segment_id = segment_id1+segment_id2
            current_segment_id = self.sequence_padding(current_segment_id)
            segment_id.append(current_segment_id)
            current_mask_id = self.sequence_padding(current_mask_id)
            mask_id.append(current_mask_id)
        
        self.token_id = token_id
        self.segment_id = segment_id
        self.mask_id = mask_id
        r"""
        print('token_id = ')
        print(token_id[0:10])
        print('segment_id = ')
        print(segment_id[0:10])
        print('mask_id = ')
        print(mask_id[0:10])
        """
        self.tensors = [torch.tensor(self.token_id,dtype=torch.long),
                       torch.tensor(self.segment_id,dtype=torch.long),
                       torch.tensor(self.mask_id,dtype=torch.long),
                       torch.tensor(self.labels,dtype=torch.long)]
    
    def __len__(self):
        return len(self.text1)
    
    def __getitem__(self,index):
        return tuple(tensor[index] for tensor in self.tensors)
    
    def sequence_padding(self,inputs,padding = 0):
        length = self.maxlen
        if len(inputs) > length:
            inputs = inputs[:length]
        outputs = []
        pad_width = (0,length-len(inputs))
        x = np.pad(inputs,pad_width,'constant',constant_values=padding)
        return x
    
dataset = ClassificationDataset(text1,text2,labels,maxlen=128)
data_iter = DataLoader(dataset,batch_size=64,shuffle=True)

bert_bin_dir="/home/xiaoguzai/数据集/bert-uncased-pytorch/"
bert_bin_file = bert_bin_dir + "pytorch_model.bin"
bert_config_file = bert_bin_dir + "bert_config.json"
tokenizer = FullTokenizer(vocab_file = '/home/xiaoguzai/数据集/bert-uncased-pytorch/vocab.txt')
bert = Bert(config)
class ClassificationModel(nn.Module):
    def __init__(self,model,config,n_labels):
        super(ClassificationModel,self).__init__()
        self.model = bert
        self.fc1 = nn.Linear(config.embedding_size,config.embedding_size)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(config.embedding_size,n_labels)
    
    def forward(self,input_ids,segment_ids,input_mask):
        #forward中传入的三个参数与return tuple(tensor[index] for tensor in self.tensors)
        #之中传入的参数相对应
        outputs = self.model(input_ids,segment_ids,input_mask)
        #[64,128,768]
        outputs = outputs[:,0]
        #[64,,768]
        outputs = self.fc1(outputs)
        outputs = self.activation(outputs)
        outputs = self.dropout(outputs)
        outputs = self.fc2(outputs)
        return outputs
    #之前这里少量return outputs返回值为None

from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
import numpy as np
ss = StratifiedKFold(n_splits=5,shuffle=True,random_state=2)
#建立4折交叉验证方法 查一下KFold函数的参数
text1 = np.array(text1)
text2 = np.array(text2)
labels = np.array(labels)
for train_index,test_index in ss.split(text1,labels):
    train_text1 = text1[np.array(train_index)]
    test_text1 = text1[test_index]
    train_text2 = text2[train_index]
    test_text2 = text2[test_index]
    train_labels = labels[train_index]
    test_labels = labels[test_index]
train_dataset = ClassificationDataset(train_text1,train_text2,train_labels,maxlen=128)
test_dataset = ClassificationDataset(test_text1,test_text2,test_labels,maxlen=128)
train_loader = DataLoader(train_dataset,batch_size=64,shuffle=True)
test_loader = DataLoader(test_dataset,batch_size=64)

from loader_bert import load_bert_data
from tqdm import tqdm
bert = load_bert_data(bert,bert_bin_file)
model = ClassificationModel(bert,config,2)
optimizer = torch.optim.Adam(model.parameters(),lr=0.00001)
loss_func = torch.nn.CrossEntropyLoss()
for epoch in range(10):
    print('epoch {}'.format(epoch+1))
    train_loss = 0
    train_acc = 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.train()
    model = model.to(device)
    model = nn.DataParallel(model)
    for batch_token_ids,batch_segment_ids,batch_mask_ids,batch_labels in tqdm(train_loader):
        batch_token_ids = batch_token_ids.to(device)
        batch_segment_ids = batch_segment_ids.to(device)
        batch_mask_ids = batch_mask_ids.to(device)
        batch_labels = batch_labels.to(device)
        output = model(batch_token_ids,batch_segment_ids,batch_mask_ids)
        loss = loss_func(output,batch_labels)
        train_loss += loss
        pred = torch.max(output, 1)[1]
        train_correct = (pred == batch_labels).sum()
        train_acc += train_correct
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    #print('Train Loss: {:.6f}, Acc: {:.6f}'.format(train_loss / (len(
    #    batch_token_ids)), train_acc / (len(train_data))))
    print('Train Loss: {:.6f}, Acc: {:.6f}'.format(train_loss, train_acc/len(train_dataset)))
    
    torch.cuda.empty_cache() 
    model.eval()
    eval_loss = 0.
    eval_acc = 0.
    for batch_token_ids,batch_segment_ids,batch_mask_ids,batch_labels in test_loader:
        
        batch_token_ids = batch_token_ids.to(device)
        batch_segment_ids = batch_segment_ids.to(device)
        batch_mask_ids = batch_mask_ids.to(device)
        batch_labels = batch_labels.to(device)
        with torch.no_grad():
            output = model(batch_token_ids,batch_segment_ids,batch_mask_ids)
        loss = loss_func(output, batch_labels)
        eval_loss += loss
        pred = torch.max(output, 1)[1]
        num_correct = (pred == batch_labels).sum()
        eval_acc += num_correct
    print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss, eval_acc/len(test_dataset)))
