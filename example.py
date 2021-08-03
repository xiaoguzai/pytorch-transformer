import torch
from tokenization import FullTokenizer
#from bertmodels import Bert
from bertmodels import Bert,Config
import json
bert_bin_dir="/home/xiaoguzai/数据集/bert-uncased-pytorch/"
bert_bin_file = bert_bin_dir + "pytorch_model.bin"
bert_config_file = bert_bin_dir + "bert_config.json"
tokenizer = FullTokenizer(vocab_file = '/home/xiaoguzai/数据集/bert-uncased-pytorch/vocab.txt')
from loader_bert import load_bert_data
config = Config(vocab_size=30522,with_pooler=False)
print(config.vocab_size)
bert = Bert(config)
#input_ids = torch.tensor([[[1,2,3,4,5]],[[0,0,0,0,0]]])
token_id1 = tokenizer.tokenize('Replace me by any text you\'d like.')
token_id1 = ["[CLS]"]+token_id1+["[SEP]"]
token_id1 = tokenizer.convert_tokens_to_ids(token_id1)
token_id1 = torch.tensor([[ 101, 5672, 2033, 2011, 2151, 3793, 2017, 1005, 1040, 2066, 1012,  102]])

bert.eval()
output_ids = bert(token_id1)
print('output_id2 = ')
print(output_ids)
r"""
tensor([[[ 0.1386,  0.1583, -0.2967,  ..., -0.2708, -0.2844,  0.4581],
         [ 0.5364, -0.2327,  0.1754,  ...,  0.5540,  0.4981, -0.0024],
         [ 0.3002, -0.3475,  0.1208,  ..., -0.4562,  0.3288,  0.8773],
         ...,
         [ 0.3799,  0.1203,  0.8283,  ..., -0.8624, -0.5957,  0.0471],
         [-0.0252, -0.7177, -0.6950,  ...,  0.0757, -0.6668, -0.3401],
         [ 0.7535,  0.2391,  0.0717,  ...,  0.2467, -0.6458, -0.3213]]]
"""