import pandas as pd
import itertools
import json
text1,text2,labels = [],[],[]
r"""
f = open('/home/xiaoguzai/数据集/第五届“达观杯” 基于大规模预训练模型的风险事件标签识别数据集/try.json')
line = f.readline()
data = ['[']
while line:
    data.append(line)
    line = f.readline()
f.close()
data.append(']')

f = open('/home/xiaoguzai/数据集/第五届“达观杯” 基于大规模预训练模型的风险事件标签识别数据集/new_try.json','w')
for index in range(len(data)):
    if index == 0:
        f.write(data[index]+'\n')
        #[+\n
    elif index == len(data)-2:
        f.write(data[index])
    elif index == len(data)-1:
        #']'
        f.write(data[index])
    else:
        f.write(data[index][:-1]+','+'\n')
f.close()
"""

title_data = []
content_data = []
with open('/home/xiaoguzai/数据集/第五届“达观杯” 基于大规模预训练模型的风险事件标签识别数据集/new_try.json','r') as load_f:
    load_dict = json.load(load_f)
    for data in load_dict:
        title_data.append(data['title'])
        content_data.append(data['content'])
        #total_data.append(data['title'])
        #total_data.append(data['content'])
#content_data长度截取:3000,title_data长度截取:150

new_content_data = []
for data in content_data:
    print(len(data))

import json
from bertmodels import Bert,Config
vocab_file = r'/home/xiaoguzai/代码/第五届“达观杯” 基于大规模预训练模型的风险事件标签识别/bert-base-count3-HMCN-F/pretrain/bert_model/vocab.txt'
vocab_size = len(open(vocab_file,'r').readlines()) 
with open('/home/xiaoguzai/数据集/bert-uncased-pytorch/config.json','r',encoding='utf8')as fp:
    json_data = json.load(fp)
json_data['vocab_size'] = vocab_size
print('json_data = ')
print(json_data)
config = Config(**json_data)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from tokenization import FullTokenizer
import numpy as np
vocab_file = "/home/xiaoguzai/代码/第五届“达观杯” 基于大规模预训练模型的风险事件标签识别/bert-base-count3-HMCN-F/pretrain/bert_model/vocab.txt"
tokenizer = FullTokenizer(vocab_file=vocab_file)
#预训练程序不需要大改，同一个程序不同的方法实现预训练
from nezha_pretraining import nezha_pretraining_store_id_data

def sequence_padding(inputs,maxlen,padding = 0):
    length = maxlen
    pad_width = [(0,0) for _ in np.shape(inputs[0])]
    #print('pad_width = ')
    #print(pad_width)
    outputs = []
    for x in inputs:
        #print('x = ')
        #print(x)
        x = x[:length]
        pad_width[0] = (0,length-len(x))
        x = np.pad(x,pad_width,'constant',constant_values=padding)
        outputs.append(x)
    return np.array(outputs)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class ClassificationDataset(Dataset):
    def __init__(self,text,maxlen):
        self.text = text
        self.maxlen = maxlen
        token_data,token_id,segment_id,mask_id = [],[],[],[]
        #sequence填充可以最后统一实现
        for index in tqdm(range(len(self.text))):
        #!!!这里构成ClassificationDataset之中的tqdm还可以优化并行操作
            current_text = text[index]
            current_token = tokenizer.tokenize(current_text)
            current_token = ["[CLS]"]+current_token
            current_id = tokenizer.convert_tokens_to_ids(current_token)
            if len(current_token) > maxlen-1:
                current_token = current_token[:maxlen-1]
                current_id = current_id[:maxlen-1]
            current_token = current_token+["[SEP]"]
            current_id = current_id+tokenizer.convert_tokens_to_ids(["[SEP]"])
            #current_segment_id = [0]*len(current_id)
            #current_mask_id = [1]*len(current_id)
            
            token_data.append(current_token)
            token_id.append(current_id)
            #segment_id.append(current_segment_id)
            #mask_id.append(current_mask_id)
        #self.segment_id = segment_id
        #self.mask_id = mask_id
        begin = 5
        self.input_id,self.label = nezha_pretraining_store_id_data(token_id,token_data,vocab_file,begin,vocab_size)
        
        self.input_id = sequence_padding(self.input_id,maxlen)
        #self.segment_id = sequence_padding(self.segment_id,maxlen)
        #self.mask_id = sequence_padding(self.mask_id,maxlen)
        self.label = sequence_padding(self.label,maxlen)
        #self.tensors = [torch.tensor(self.input_id,dtype=torch.long).to(device),
        #               torch.tensor(self.label,dtype=torch.long).to(device)]
        #self.tensors = [torch.tensor(self.input_id,dtype=torch.long).to(device),
        #                torch.tensor(self.label,dtype=torch.long).to(device)]
        self.tensors = [torch.tensor(self.input_id,dtype=torch.long).to(device),
                       torch.tensor(self.label,dtype=torch.long).to(device)]
        #tensor之后，清除原先的内存会导致无法训练，注意这里暂时不能够放入device之中
        
    def __len__(self):
        return len(self.tensors[0])
    
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
r"""
title_dataset = ClassificationDataset(title_data,maxlen=150)
title_data_iter = DataLoader(title_dataset,batch_size=2,shuffle=True)

content_maxlen = 500
content_dataset = ClassificationDataset(content_data,maxlen=maxlen)
content_data_iter = DataLoader(content_dataset,batch_size=20,shuffle=True)
"""

from bertmodels import Bert
from tqdm import tqdm
import gc
config.with_mlm = True
model = Bert(config)
optimizer = torch.optim.AdamW(model.parameters())
loss_func = torch.nn.CrossEntropyLoss()
#torch.cuda.set_device(0)
model = model.to(device)
model = nn.DataParallel(model)
model.train()
scaler = torch.cuda.amp.GradScaler()
#content_maxlen = 500
content_maxlen = 200
for epoch in range(1):
    print('epoch {}'.format(epoch+1))
    train_loss = 0
    train_acc = 0
    for filenum in range(0,500):
        title_data = []
        content_data = []
        filename = '/home/xiaoguzai/数据集/第五届“达观杯” 基于大规模预训练模型的风险事件标签识别数据集/data/unlabeled_data'+str(filenum)+'.json'
        with open(filename,'r') as load_f:
            load_dict = json.load(load_f)
            print('load file %s'%filename)
            for data in tqdm(load_dict):
                title_data.append(data['title'])
                currentcontent = data['content']
                currentlen = len(currentcontent)
                if currentlen%content_maxlen == 0:
                    totalnumber = currentlen//content_maxlen
                else:
                    totalnumber = currentlen//content_maxlen
                    totalnumber = totalnumber+1
                for number in range(totalnumber):
                    right = min(currentlen,(number+1)*content_maxlen)
                    content_data.append(currentcontent[number*content_maxlen:right])
            print('load %s finish'%filename)
        
        print('begin %s title_dataset'%filename)
        title_dataset = ClassificationDataset(title_data,maxlen=150)
        title_dataset_len = len(title_dataset)
        title_data_iter = DataLoader(title_dataset,batch_size=128,shuffle=True)
        #batch_size = 256
        print('title_dataset finish')
        print('*********************begin train**********************')
        train_loss = 0
        train_acc = 0
        for batch_token_ids,batch_labels in tqdm(title_data_iter,colour='blue'):
            batch_token_ids = batch_token_ids.to(device)
            batch_labels = batch_labels.to(device)
            output = model(batch_token_ids)
            output = output.view(-1,vocab_size)
            batch_labels = batch_labels.view(-1)
            loss = loss_func(output,batch_labels)
            train_loss += loss
            pred = torch.max(output, 1)[1]
            train_correct = (pred == batch_labels).sum()
            train_acc += train_correct
            optimizer.zero_grad()
            #loss.backward()
            #optimizer.step()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            #使用混合精度进行预训练
        print('Train Loss: {:.6f}, Acc: {:.6f}'.format(train_loss/title_dataset_len, train_acc/title_dataset_len))
        print('********************finish train***********************')
        
        print('begin %s content_dataset'%filename)
        content_dataset = ClassificationDataset(content_data,maxlen=content_maxlen)
        content_data_iter = DataLoader(content_dataset,batch_size=64,shuffle=True)
        content_dataset_len = len(content_dataset)
        print('content_dataset finish')
        print('####################begin train########################')
        train_loss = 0
        train_acc = 0
        content_batch_size = 64
        step = 0
        r"""
        for batch_token_ids,batch_labels in tqdm(content_dataset,colour='blue'):
            #数据较大的时候拿出一个批次再放入gpu之中,减少显存的消耗
            step = step+1
            current_batch_token_ids = batch_token_ids.tolist()
            current_batch_labels = batch_labels.tolist()
            batch_token_ids = torch.tensor([current_batch_token_ids],dtype=torch.long).to(device)
            batch_labels = torch.tensor([current_batch_labels],dtype=torch.long).to(device)
            output = model(batch_token_ids)
            output = output.view(-1,vocab_size)
            batch_labels = batch_labels.view(-1)
            loss = loss_func(output,batch_labels)
            train_loss += loss
            pred = torch.max(output, 1)[1]
            train_correct = (pred == batch_labels).sum()
            train_acc += train_correct
            scaler.scale(loss).backward()
            #反向传播，计算当前的梯度(会叠加),因为会叠加
            #所以才需要scaler.zero_grad()清空过往的梯度
            del(current_batch_token_ids)
            del(current_batch_labels)
            gc.collect()
            if (step+1)%content_batch_size == 0:
            #使用梯度累积，节约显存
            #optimizer.zero_grad()
            #scaler.scale(loss).backward()
            #scaler.step(optimizer)
            #scaler.update()
                scaler.step(optimizer)
                #根据梯度更新网络参数
                optimizer.zero_grad()
                #清空过往的梯度
                scaler.update()
        """
        train_loss = 0
        train_acc = 0
        for batch_token_ids,batch_labels in tqdm(content_data_iter,colour='blue'):
            batch_token_ids = batch_token_ids.to(device)
            batch_labels = batch_labels.to(device)
            output = model(batch_token_ids)
            output = output.view(-1,vocab_size)
            batch_labels = batch_labels.view(-1)
            loss = loss_func(output,batch_labels)
            train_loss += loss
            pred = torch.max(output, 1)[1]
            train_correct = (pred == batch_labels).sum()
            train_acc += train_correct
            optimizer.zero_grad()
            #loss.backward()
            #optimizer.step()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        print('Train Loss: {:.6f}, Acc: {:.6f}'.format(train_loss/content_dataset_len, train_acc/content_dataset_len))
        torch.cuda.empty_cache() 
        print('####################finish train########################')
        if filenum%50 == 0:
            torch.save(model.state_dict(),'/home/xiaoguzai/数据集/第五届“达观杯” 基于大规模预训练模型的风险事件标签识别数据集/filenum='+str(filenum)+'-'+'model_epoch='+str(epoch)+'.pth')
torch.save(model.state_dict(),'/home/xiaoguzai/数据集/第五届“达观杯” 基于大规模预训练模型的风险事件标签识别数据集/model_epoch='+str(epoch)+'.pth')

