from t5tokenization import T5Tokenizer
from mt5 import MT5,MT5Config,MT5Generation,greedy_generate
from loader_mt5 import load_mt5_model_data,load_mt5_generation_data
import torch
#input_ids = torch.tensor([[1,2,3,4,5],[1,2,3,4,5]])
input_ids = torch.tensor([[1,2,3,0,0],[1,2,3,0,0]])
config = MT5Config()
t5tokenizer = T5Tokenizer(config,'/home/xiaoguzai/模型/mt5/spiece.model')
result_id = t5tokenizer.get_input_ids("I love you forever.")
print('result_id = ')
print(result_id)