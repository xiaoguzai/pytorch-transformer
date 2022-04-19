from mt5 import MT5,MT5Config
from loader_mt5 import load_mt5_data
import torch
input_ids = torch.tensor([[1,2,3,4,5]])
config = MT5Config()
print('config.dropout_rate = ')
print(config.dropout_rate)
model = MT5(config)
#model.eval()
model = load_mt5_data(model,'/home/xiaoguzai/模型/mt5/pytorch_model.bin')
model.eval()
result = model(input_ids)
print('result = ')
print(result)