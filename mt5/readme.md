调用贪婪搜索代码部分
```python
from mt5 import MT5,MT5Config
from loader_mt5 import load_mt5_model_data
from mt5 import MT5Generation
from loader_mt5 import load_mt5_generation_data
from mt5 import greedy_generate
import torch
input_ids = torch.tensor([[1,2,3,4,5],[1,2,3,4,5]])
config = MT5Config()
#model = MT5(config)
#model.eval()
#model = load_mt5_model_data(model,'/home/xiaoguzai/模型/mt5/pytorch_model.bin')
model = MT5Generation(config)
model.eval()
model = load_mt5_generation_data(model,'/home/xiaoguzai/模型/mt5/pytorch_model.bin')
greedy_generate(model,config,input_ids,labels=None,max_length=20)
```
调用计算loss损失函数部分
```python
from mt5 import MT5,MT5Config
from loader_mt5 import load_mt5_model_data,load_mt5_generation_data
from mt5 import MT5Generation,greedy_generate
import torch
import random
import numpy as np
import os
RANDOM_SEED = 42 # any random number
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) # CPU
    torch.cuda.manual_seed(seed) # GPU
    torch.cuda.manual_seed_all(seed) # All GPU
    os.environ['PYTHONHASHSEED'] = str(seed) # 禁止hash随机化
    torch.backends.cudnn.deterministic = True # 确保每次返回的卷积算法是确定的
    torch.backends.cudnn.benchmark = False # True的话会自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。False保证实验结果可复现
set_seed(RANDOM_SEED)
input_ids = torch.tensor([[486,250099,12747,263,281,250098,10676,1],\
                          [486,250099,12747,263,281,250098,10676,1]])
decoder_input_ids = torch.tensor([[250099,64712,10990,250098,287,250097,1,0,0,0],\
                                  [250099,64712,10990,250098,287,250097,1,0,0,0]])
#decoder_input_ids = torch.tensor([[250099,64712,10990,250098,287,250097,1],\
#                                  [250099,64712,10990,250098,287,250097,1]])
config = MT5Config()
print('config.dropout_rate = ')
print(config.dropout_rate)
model = MT5Generation(config)

#model = load_mt5_model_data(model,'/home/xiaoguzai/模型/mt5/pytorch_model.bin')
model = load_mt5_generation_data(model,'/home/xiaoguzai/模型/mt5/pytorch_model.bin')
#model.train()
model.eval()
print('111input_ids = 111')
print(input_ids)
print('111111111111111111')
print('decoder_input_ids = ')
print(decoder_input_ids)
print('====================')
result,loss, = model(input_ids=input_ids,labels=decoder_input_ids,generate=False)
print('@@@loss = @@@')
print(loss)
print('@@@@@@@@@@@@@')
```
