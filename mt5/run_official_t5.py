import torch
from transformers import MT5Tokenizer, MT5ForConditionalGeneration
tokenizer = MT5Tokenizer.from_pretrained('/home/xiaoguzai/模型/mt5')
model = MT5ForConditionalGeneration.from_pretrained('/home/xiaoguzai/模型/mt5')
input_ids = tokenizer("summarize: studies have shown that owning a dog is good for you", return_tensors="pt").input_ids  # Batch size 1
#print('input_ids = ')
#print(input_ids)
input_ids = torch.tensor([[1,2,3,4,5]])
outputs = model.generate(input_ids)
print('outputs = ')
print(outputs)