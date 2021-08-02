import torch
from tokenization import FullTokenizer
#from bertmodels import Bert
from bertmodels import Bert
inputs = torch.tensor([[[1,2,3,4,5]],[[0,0,0,0,0]]])
bertmodel = Bert()
print(bertmodel)
outputs = bertmodel(inputs)
print('...outputs = ...')
print(outputs)
