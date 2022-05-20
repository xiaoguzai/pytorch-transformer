测试代码部分

计算loss

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
input_ids = torch.tensor([[486,250099,12747,263,281,250098,10676,1,0,0,0],\
                          [486,250099,12747,263,281,250098,10676,1,0,0,0]])
decoder_input_ids = torch.tensor([[250099,64712,10990,250098,287,250097,1,-100,-100,-100],\
                                  [250099,64712,10990,250098,287,250097,1,-100,-100,-100]])
#decoder_input_ids用-100进行填充
config = MT5Config()
model = MT5Generation(config)
model = load_mt5_generation_data(model,'/home/xiaoguzai/模型/mt5/pytorch_model.bin')
model.eval()
result,loss = model(input_ids=input_ids,labels=decoder_input_ids,generate=False)
```

得到的loss内容

```
loss = tensor(4.2771, grad_fn=<NllLossBackward)
```

常规的贪婪搜索生成部分

```python
from mt5 import MT5,MT5Config
from loader_mt5 import load_mt5_model_data
from mt5 import MT5Generation
from loader_mt5 import load_mt5_generation_data
from mt5 import greedy_generate,new_greedy_generate
import torch
input_ids = torch.tensor([[1,2,3,4,5],[1,2,3,4,5]])
config = MT5Config()
model = MT5Generation(config)
model.eval()
model = load_mt5_generation_data(model,'/home/xiaoguzai/模型/mt5/pytorch_model.bin')
output_ids = new_greedy_generate(model,config,input_ids,labels=None,max_length=20)
print(output_ids)
```

得到的output_ids的内容

```
output_ids = 
tensor([[250099,  80895,   6708,    294,  21475,    263,    262, 250098,    298,
         250097,      1],
        [250099,  80895,   6708,    294,  21475,    263,    262, 250098,    298,
         250097,      1]])
```

优化后的贪婪搜索生成部分

```python
from mt5 import MT5,MT5Config
from loader_mt5 import load_mt5_model_data
from mt5 import MT5Generation
from loader_mt5 import load_mt5_generation_data
from mt5 import greedy_generate
import torch
input_ids = torch.tensor([[1,2,3,0,0],[1,2,3,0,0]])
config = MT5Config()
model = MT5Generation(config)
model = load_mt5_generation_data(model,'/home/xiaoguzai/模型/mt5/pytorch_model.bin')
model.eval()
result_ids = greedy_generate(model,config,input_ids)
```

得到的result_ids的内容

```
result_ids = 
tensor([[250099,    259, 250098,    259, 250097,    259, 250096,    259, 250095,
            259, 250094,    259, 250093,    259, 250092,    259, 250091,    259,
         250090,    259],
        [250099,    259, 250098,    259, 250097,    259, 250096,    259, 250095,
            259, 250094,    259, 250093,    259, 250092,    259, 250091,    259,
         250090,    259]])
```

