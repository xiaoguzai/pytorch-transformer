## longformer调用测试代码
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
input_ids = torch.ones(2,1025).long()
output_ids = longformermodel(input_ids)
print('output_id2 = ')
print(output_ids)
```
