安装
```
pip install pythonicforbert
```
一个简单的例子
```python
import torch
#from tokenization import FullTokenizer
#from bertmodels import Bert
from pythonicforbert import FullTokenizer
from pythonicforbert import Nezha,NezhaConfig

from pythonicforbert import get_model_function
import json
bert_bin_dir="/home/xiaoguzai/模型/bert-base/"

#bert_bin_file = bert_bin_dir + "pytorch_model.bin"
bert_bin_file = '/home/xiaoguzai/模型/bert-wwm/pytorch_model.bin'
bert_config_file = bert_bin_dir + "config.json"
tokenizer = FullTokenizer(vocab_file = bert_bin_dir+'vocab.txt')
bertmodel,bertconfig,get_data = get_model_function('bert-base')

with open(bert_config_file,'r',encoding='utf8')as fp:
    json_data = json.load(fp)
print(json_data)

config = bertconfig(**json_data)
bert = bertmodel(config)
bertmodel = get_data(bert,bert_bin_file)
r"""
test(**kwargs)** 的作用则是把字典 kwargs 变成关键字参数传递。
比如上面这个代码，如果 kwargs 等于 {'a':1,'b':2,'c':3} ，那这个代码就等价于 test(a=1,b=2,c=3) 
"""
token_id1 = tokenizer.tokenize('Replace me by any text you\'d like.')
token_id1 = ["[CLS]"]+token_id1+["[SEP]"]
token_id1 = tokenizer.convert_tokens_to_ids(token_id1)
token_id1 = torch.tensor([[ 101, 5672, 2033, 2011, 2151, 3793, 2017, 1005, 1040, 2066, 1012,  102]])

bert.eval()
output_ids = bert(token_id1)
print('output_id2 = ')
print(output_ids)
```
使用pythoicforbert调用longformer模型的例子
```python
import torch
#from tokenization import FullTokenizer
#from bertmodels import Bert
from pythonicforbert import FullTokenizer
from pythonicforbert import Nezha,NezhaConfig

from pythonicforbert import get_model_function
import json
longformer_bin_file = '/home/xiaoguzai/模型/Longformer/pytorch_model.bin'
longformer_config_file = '/home/xiaoguzai/模型/Longformer/config.json'
LongFormerModel,LongFormerConfig,get_data = get_model_function('longformer-base')

import json
with open('/home/xiaoguzai/模型/Longformer/config.json','r',encoding='utf8')as fp:
    json_data = json.load(fp)

longformerconfig = LongFormerConfig(**json_data)
longformer = LongFormerModel(longformerconfig)
longformermodel = get_data(longformer,longformer_bin_file)
#bert.eval()
longformermodel.eval()

output_ids = longformermodel(torch.tensor([[1,2,3,4,5],[1,2,3,4,5]]))
print('output_id2 = ')
print(output_ids)
```
