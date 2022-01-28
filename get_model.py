from .bert import BertConfig,Bert,load_bert_data
from .nezha import NezhaConfig,Nezha,load_nezha_data
from .roberta import RobertaConfig,Roberta,load_roberta_data
from .longformer import LongFormerConfig,LongFormer,load_longformer_data
from .loader_pretrain_weights import load_pretrain_data
from .tokenization import FullTokenizer
def get_model_function(name):
    name_list = name.split('-')
    name_list[0] = name_list[0].lower()
    name_list[1] = name_list[1].lower()
    #bert-base,nezha-base,bert-pretrain
    
    if name_list[0] == 'bert':
        model,config = Bert,BertConfig
        read_data_function = load_bert_data
    elif name_list[0] == 'nezha':
        model,config = Nezha,NezhaConfig
        read_data_function = load_nezha_data
    elif name_list[0] == 'roberta':
        model,config = Roberta,RobertaConfig
        read_data_function = load_roberta_data
    elif name_list[0] == 'longformer':
        model,config = LongFormer,LongFormerConfig
        read_data_function = load_longformer_data
    else:
        print('目前暂时未有%s模型,请等待更新🙏🙏🙏'%(name))

    if name_list[1] == 'pretrain':
        read_data_function = load_pretrain_data
    return model,config,read_data_function
	
	
