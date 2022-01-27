import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import math
import logging
#注意：使用英文的时候再调用roberta，使用中文的时候调用bert即可
class LongFormerConfig(object):
    def __init__(self,
                initializer_range=0.02,#1
                embedding_size=768,#4
                project_embeddings_with_bias=True,#5
                vocab_size=50265,#6
                hidden_dropout=0.1,#10
                extra_tokens_vocab_size=None,#11
                project_position_embeddings=True,#12
                mask_zero=False,#13
                adapter_size=None,#14
                hidden_act='gelu',#15
                adapter_init_scale=0.001,#16
                num_attention_heads=12,#17
                size_per_head=None,#18
                attention_probs_dropout_prob=0.1,#22
                negative_infinity=-10000.0,#23
                intermediate_size=3072,#24
                intermediate_activation='gelu',#25
                num_layers=12,#26
                #获取对应的切分分割字符内容
                directionality = 'bidi',
                pooler_fc_size = 768,
                hidden_size = 1024,
                pooler_num_attention_heads = 12,
                pooler_num_fc_layers = 3,
                pooler_size_per_head = 128,
                pooler_type = "first_token_transform",
                type_vocab_size = 1,
                with_prediction = False,
                mlm_activation = 'softmax',
                attention_mode = 'longformer',
                #max_relative_position = 512,
                solution = 'seq2seq',
                max_relative_position = 64,
                with_pooler = False,
                max_position_embeddings = 4098,
                layer_norm_eps = 1e-05,
                num_hidden_layers = 12,
                attention_window = [512,512,512,512,512,512,512,512,512,512,512,512],
                bos_token_id = 0,
                eos_token_id = 2,
                gradient_checkpointing = False,
                hidden_dropout_prob = 0.1,
                ignore_attention_mask = False,
				 
                pad_token_id = 1,
                sep_token_id = 2,
                *args, **kwargs):
        self.initializer_range = initializer_range#1
        self.embedding_size = embedding_size#4
        self.embedding_size = hidden_size
        self.project_embeddings_with_bias = project_embeddings_with_bias#5
        self.vocab_size = vocab_size#6
        self.token_type_vocab_size = type_vocab_size#9
        #roberta之中与bert不同的地方：因为roberta去除了下一个句子的预测，所以position_embeddings全零
        self.hidden_dropout = hidden_dropout_prob#10
        self.extra_tokens_vocab_size = extra_tokens_vocab_size#11
        self.project_position_embeddings = project_position_embeddings#12
        self.mask_zero = mask_zero#13
        self.adapter_size = adapter_size#14
        self.adapter_init_scale = adapter_init_scale#16
        self.num_attention_heads = num_attention_heads#17注意力头数，需指定
        assert hidden_size%num_attention_heads == 0,"size_per_head必须能够整除num_attention_heads"
        self.size_per_head = hidden_size//num_attention_heads#18
        self.attention_probs_dropout_prob = attention_probs_dropout_prob#22
        self.negative_infinity = negative_infinity#23
        self.intermediate_size = intermediate_size#24
        self.intermediate_activation = intermediate_activation#25
        self.num_layers = num_layers#26 attention层数，需指定
        self.directionality = directionality
        self.pooler_fc_size = pooler_fc_size
        self.pooler_num_attention_heads = pooler_num_attention_heads
        self.pooler_num_fc_layers = pooler_num_fc_layers
        self.pooler_size_per_head = pooler_size_per_head
        self.pooler_type = pooler_type
        self.mlm_activation = mlm_activation
        self.attention_mode = attention_mode
        self.solution = solution
        self.max_position_embeddings = max_position_embeddings
        self.max_relative_position = max_relative_position
        self.with_pooler = with_pooler
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.num_layers = num_hidden_layers
        self.with_mlm = with_prediction
        self.with_prediction = with_prediction
        
        self.attention_window = attention_window
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.gradient_checkpointing = gradient_checkpointing
        
        self.pad_token_id = pad_token_id
        self.sep_token_id = sep_token_id
        
        self.ignore_attention_mask = ignore_attention_mask

class LongFormer(nn.Module):
    def __init__(self,config):
        #这里初步先将所有的参数都放入__init__之中
        #后期可以将不需要更改的参数放入build函数之中
        #之前看的内容相当于对params更新name，并且将所有属性中带有name的内容进行更新
        #这样就成功定义了name的对应值
        #super(Nezha, self).__init__()
        super(LongFormer,self).__init__()
        self.config = config
        self.longformerembeddings = Embeddings(config)
        self.longformer_encoder_layer = nn.ModuleList()
        for layer_ndx in range(config.num_layers):
            encoder_layer = Transformer(config,layer_ndx)
            self.longformer_encoder_layer.append(encoder_layer)
        if config.with_pooler:
            self.longformer_pooler = nn.Linear(config.embedding_size,config.embedding_size)
        if config.with_prediction:
            self.prediction_dense0 = nn.Linear(config.embedding_size,config.embedding_size)
            self.prediction_norm = nn.LayerNorm(config.embedding_size,eps=1e-12)
            self.prediction_dense1 = nn.Linear(config.embedding_size,config.vocab_size)
        #print('self.config.initializer_range = ')
        #print(self.config.initializer_range)
        
    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            #print('888self.config.initializer_range = 888')
            #print(self.config.initializer_range)
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            #一般这里的mean=0.0,stddev标准差为1
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    
    def _pad_to_window_size(
        self,
        input_ids,
        segment_ids,
        attention_mask):
        #将输入的tensor内容padding成为能够达到window_size的tensor内容
        if isinstance(self.config.attention_window,int):
            attention_window = self.config.attention_window
        else:
            attention_window = max(self.config.attention_window)
        #attention_window = 512
        batch_size,seq_len = input_ids.shape[:2]
        padding_len = (attention_window-seq_len%attention_window)%attention_window
        if padding_len > 0:
            logging.info(
                f"Input ids are automatically padded from {seq_len} to {seq_len + padding_len} to be a multiple of "
                f"`config.attention_window`: {attention_window}"
            )
            input_ids = nn.functional.pad(input_ids,(0,padding_len),value=self.config.pad_token_id)
            attention_mask = nn.functional.pad(attention_mask,(0,padding_len),value=False)
            segment_ids = nn.functional.pad(segment_ids,(0,padding_len),value=0)
        return padding_len,input_ids,segment_ids,attention_mask
    
    
    def forward(self,
                input_ids,
                segment_ids = None,
                attention_mask = None,
                global_attention_mask = None,
                ):
        r"""
        global_attention_mask:Mask to decide the attention given on each token,local attention or
        global attention.Tokens with global attention attends to all other tokens,and all other 
        tokens attend to them.
        """
        if segment_ids == None:
            segment_ids = torch.zeros_like(input_ids)
        device = input_ids.device
        
        if attention_mask is None:
            attention_mask = torch.ones(input_ids.size(),device=device)
        #当输入的内容input_ids = torch.tensor([[1,2,3,4,5]])的时候，attention_mask = tensor([[1,1,1,1,1]]),
        #segment_ids = tensor([[0,0,0,0,0]])
        
        #融合attention_mask和global_attention_mask的内容
        if global_attention_mask is not None:
            if attention_mask is not None:
                attention_mask = attention_mask*(global_attention_mask+1)
            else:
                attention_mask = global_attention_mask+1
        #这里的global_attention_mask为None，所以attention_mask = tensor([[1,1,1,1,1]])
        
        padding_len,input_ids,segment_ids,attention_mask = self._pad_to_window_size(input_ids,segment_ids,attention_mask)
        #此处有个_pad_to_window_size，如果长度不够512的情况下补充到512的长度
        input_shape = input_ids.shape
        extended_attention_mask = attention_mask[:,None,None,:]
        extended_attention_mask = (1.0-extended_attention_mask)*(-10000.0)
        extended_attention_mask = extended_attention_mask[:,0,0,:]
        is_index_masked = extended_attention_mask < 0
        is_index_global_attn = extended_attention_mask > 0
        is_global_attn = is_index_global_attn.flatten().any().item()
        
        outputs = self.longformerembeddings(input_ids,segment_ids)
        for layer_ndx in self.longformer_encoder_layer:
            #if self.config.gradient_checkpointing == True:调用时间换取显存
            outputs = layer_ndx(outputs,extended_attention_mask,is_index_masked)
            
        if self.config.with_pooler:
            outputs = self.roberta_pooler(outputs)
        if self.config.with_prediction:
            outputs = self.prediction_dense0(outputs)
            outputs = self.prediction_norm(outputs)
            outputs = self.prediction_dense1(outputs)
        return outputs

class LayerNorm(nn.Module):
    "A layernorm module in the TF style (epsilon inside the square root)."
    def __init__(self, embedding_size, variance_epsilon=1e-12):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(embedding_size))
        self.beta  = nn.Parameter(torch.zeros(embedding_size))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        #keep_dim=True:保持输出的维度,keepdim=False:输出在求范数的维度上
        #元素个数变为1
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta

class Embeddings(nn.Module):
    def __init__(self,config):
        #之前__init__之中少写了一个self,报错multiple initialize
        #先能跑通一个网络层，再往里面加入网络层
        super(Embeddings, self).__init__()
        self.word_embeddings_layer = nn.Embedding(config.vocab_size,config.embedding_size)
        self.segment_embeddings_layer = nn.Embedding(config.token_type_vocab_size,config.embedding_size)
        self.position_embeddings_layer = nn.Embedding(config.max_position_embeddings,config.embedding_size)
        self.config = config
        self.layer_normalization = nn.LayerNorm(config.embedding_size,eps=config.layer_norm_eps)
        #这里roberta,bert等模型存在小bug，就是eps没有配置好
        self.dropout_layer = nn.Dropout(config.hidden_dropout)

    def forward(self,input_ids,segment_ids,mask_ids=None):
        if segment_ids == None:
            segment_ids = torch.zeros_like(input_ids)
        seq_len = input_ids.size(1)
        #长度为(batch_size,seq_len)
        current_mask = input_ids.ne(self.config.pad_token_id).int()
        #这里的int()类型转换很重要，否则输出的就是True或者False类型的数据,True与False的特点在于
        #True增加值仍然为True
        #input_ids = torch.tensor([[1,2,3,4,5,1,1,1,...1,1,1]])的时候，current_mask = tensor([[0,1,1,1,1,0,.....0,0,0]])
        incremental_indices = torch.cumsum(current_mask,dim=1).type_as(current_mask)*current_mask
        position_ids = incremental_indices.long()+self.config.pad_token_id
        #本身incremental_indices = tensor([[0,1,1,1,1,0,0,...0,0,0]]),加上pad_token_id之后
        #position_ids = tensor([[1,2,2,2,2,1,1,...1,1,1]])
        data1 = self.word_embeddings_layer(input_ids)
        data2 = self.segment_embeddings_layer(segment_ids)
        data3 = self.position_embeddings_layer(position_ids)
        results = self.word_embeddings_layer(input_ids)+self.segment_embeddings_layer(segment_ids)+self.position_embeddings_layer(position_ids)
        results = self.layer_normalization(results)
        results = self.dropout_layer(results)
        return results
    
    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

def gelu(x):
    "Implementation of the gelu activation function by Hugging Face"
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def get_activation(activation):
    if activation == 'gelu':
        return F.gelu
    elif activation == 'relu':
        return F.relu
    elif activation == 'tanh':
        return torch.tanh

class Transformer(nn.Module):
    def __init__(self,
                 config,
                 layer_id,
                 **kwargs):
        super(Transformer,self).__init__()
        self.attention = AttentionLayer(config,layer_id)
        self.dense0 = nn.Linear(config.embedding_size,config.embedding_size)
        self.dropout0 = nn.Dropout(config.attention_probs_dropout_prob)
        self.layer_norm0 = nn.LayerNorm(config.embedding_size,eps=1e-12)
        self.dense = nn.Linear(config.embedding_size,config.intermediate_size)
        self.activation = get_activation(config.hidden_act)
        self.dense1 = nn.Linear(config.intermediate_size,config.embedding_size)
        self.dropout1 = nn.Dropout(config.attention_probs_dropout_prob)
        self.layer_norm1 = nn.LayerNorm(config.embedding_size,eps=1e-12)
        
    
    def forward(self,inputs,masks=None,is_index_masked=None,**kwargs):
        residual = inputs
        embedding_output = inputs
        embedding_output = self.attention(inputs,masks,is_index_masked)
        embedding_output = self.dense0(embedding_output)
        embedding_output = self.dropout0(embedding_output)
        
        embedding_output = self.layer_norm0(residual+embedding_output)
        residual = embedding_output
        embedding_output = self.dense(embedding_output)
        embedding_output = self.activation(embedding_output)
        embedding_output = self.dense1(embedding_output)
        embedding_output = self.dropout1(embedding_output)
        embedding_output = self.layer_norm1(residual+embedding_output)
        
        return embedding_output

def split_last(x, shape):
    "split the last dimension to given shape"
    shape = list(shape)
    assert shape.count(-1) <= 1
    if -1 in shape:
        shape[shape.index(-1)] = int(x.size(-1) / -np.prod(shape))
    return x.view(*x.size()[:-1], *shape)

def merge_last(x, n_dims):
    "merge the last n_dims to a dimension"
    s = x.size()
    assert n_dims > 1 and n_dims < len(s)
    return x.view(*s[:-n_dims], -1)

class AttentionLayer(nn.Module):
    def __init__(self,
                 config,
                 layer_id,
                 **kwargs):
        super(AttentionLayer,self).__init__()
        self.query_layer = nn.Linear(config.embedding_size,config.embedding_size)
        self.key_layer = nn.Linear(config.embedding_size,config.embedding_size)
        self.value_layer = nn.Linear(config.embedding_size,config.embedding_size)
        
        self.query_global = nn.Linear(config.embedding_size,config.embedding_size)
        self.key_global = nn.Linear(config.embedding_size,config.embedding_size)
        self.value_global = nn.Linear(config.embedding_size,config.embedding_size)
        
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.config = config
        
        self.layer_id = layer_id
        attention_window = config.attention_window[self.layer_id]
        assert (
            attention_window % 2 == 0
        ), f"`attention_window` for layer {self.layer_id} has to be an even value. Given {attention_window}"
        assert (
            attention_window > 0
        ), f"`attention_window` for layer {self.layer_id} has to be positive. Given {attention_window}"

        self.one_sided_attn_window_size = attention_window // 2
        #self.config.num_attention_heads = 12,self.config.size_per_head = 64,self.embedding_size = 768
        #self.one_sided_attn_window_size = 256
    
    def _sliding_chunks_query_key_matmul(self,query,key,one_sided_attn_window_size):
        batch_size,seq_len,num_attention_heads,size_per_head = query.size()
        query = query.transpose(1,2).reshape(batch_size*num_attention_heads,seq_len,size_per_head)
        key = key.transpose(1,2).reshape(batch_size*num_attention_heads,seq_len,size_per_head)

        #query = (512,2,12,64),transpose(0,1)之后query = (2,512,12,64),chunks_count = 1
        #transpose(1,2)之后query = (2,12,512,64),reshape之后query = (24,512,64),这样reshape的结果与transformers
        #reshape之后得到的结果一致，这里如果不先进行reshape之后而是直接进行view操作的话，形状无法直接对应上
        #这是因为view是进行铺平之后再去截取相应内容，所以如果(2,512,12,64)->(24,512,64)会铺平之后再去寻找
        #直接query.view([batch_size*self.config.num_attention_heads,1,seq_len,self.config.size_per_head])的结果无法对应上
        r"""
        query = tensor([[[0.0042,0.00697,-0.0077,...]],
                        [[-0.1489,-0.0298,0.0020,...]],
                        ..............................
                        [[-0.4297,0.2865,-0.1520,...]]])
        key = tensor([[[0.3744,-0.4979,-0.1762,...]],
                      [[0.1873,-0.5420,-0.6873,...]],
                      ..............................
                      [[-3.4673,3.4823,1.0681,...]]])
        """
        
        #query = (72,1024,64),key = (72,1024,64)
        query = query.view(
            batch_size*num_attention_heads,
            seq_len // (one_sided_attn_window_size*2),
            one_sided_attn_window_size*2,
            size_per_head,
        )
        key = key.view(
            batch_size*num_attention_heads,
            seq_len // (one_sided_attn_window_size*2),
            one_sided_attn_window_size*2,
            size_per_head,
        )
        #注意这里要加入括号，因为运算有优先级
        #query.size = tensor([72,2,512,64]),key.size = tensor([72,2,512,64])
        #在_sliding_chunks_query_key_matmul之中transpose之后得到的query111.txt
        query_size = list(query.size())
        query_size[1] = query_size[1]*2-1
        query_stride = list(query.stride())
        query_stride[1] = query_stride[1]//2
        query = query.as_strided(size=query_size,stride=query_stride)
        
        key_size = list(key.size())
        key_size[1] = key_size[1]*2-1
        key_stride = list(key.stride())
        key_stride[1] = key_stride[1]//2
        key = key.as_strided(size=key_size,stride=key_stride)
        attention_scores = torch.matmul(query,key.transpose(-1,-2))
        attention_scores = nn.functional.pad(
            attention_scores,(0,0,0,1)
        )
        attention_scores = attention_scores.view(*attention_scores.size()[:-2],attention_scores.size(-1),attention_scores.size(-2))
        diagonal_attention_scores = attention_scores.new_empty(
            (batch_size*num_attention_heads,seq_len//one_sided_attn_window_size,one_sided_attn_window_size,one_sided_attn_window_size*2+1)
        )
        
        #diagonal_attention_scores[:,:-1,:,one_sided_attn_window_size:] = attention_scores[:,:,:one_sided_attn_window_size,:one_sided_attn_window_size+1]
        diagonal_attention_scores[0:batch_size*num_attention_heads,0:seq_len//one_sided_attn_window_size-1,\
                                  0:one_sided_attn_window_size,one_sided_attn_window_size:one_sided_attn_window_size*2+1] = \
        attention_scores[0:batch_size*num_attention_heads,0:seq_len//one_sided_attn_window_size-1,0:one_sided_attn_window_size,0:one_sided_attn_window_size+1]
        
        #diagonal_attention_scores[:,-1,:,one_sided_attn_window_size:] = attention_scores[:,-1,one_sided_attn_window_size:,:one_sided_attn_window_size+1]
        diagonal_attention_scores[0:batch_size*num_attention_heads,-1,0:one_sided_attn_window_size,one_sided_attn_window_size:one_sided_attn_window_size*2+1] = \
        attention_scores[0:batch_size*num_attention_heads,-1,one_sided_attn_window_size:one_sided_attn_window_size*2,0:one_sided_attn_window_size+1]
        
        
        #diagonal_attention_scores[:,1:,:,:one_sided_attn_window_size] = attention_scores[:,:,-(one_sided_attn_window_size+1):-1,one_sided_attn_window_size+1:]
        diagonal_attention_scores[0:batch_size*num_attention_heads,1:seq_len//one_sided_attn_window_size,\
                                  0:one_sided_attn_window_size,0:one_sided_attn_window_size] = \
        attention_scores[0:batch_size*num_attention_heads,0:seq_len//one_sided_attn_window_size-1,\
                         one_sided_attn_window_size-1:one_sided_attn_window_size*2-1,one_sided_attn_window_size+1:one_sided_attn_window_size*2+1]
        
        #diagonal_attention_scores[:,0,1:one_sided_attn_window_size,1:one_sided_attn_window_size] = attention_scores[:,0,:one_sided_attn_window_size-1,1-one_sided_attn_window_size:]
        diagonal_attention_scores[0:batch_size*num_attention_heads,0,1:one_sided_attn_window_size,1:one_sided_attn_window_size] = \
        attention_scores[0:batch_size*num_attention_heads,0,0:one_sided_attn_window_size-1,one_sided_attn_window_size+2:2*one_sided_attn_window_size+1]
        #产生差异的句子
        
        diagonal_attention_scores = diagonal_attention_scores.view(batch_size,num_attention_heads,seq_len,2*one_sided_attn_window_size+1).transpose(2,1)
        
        beginning_mask_2d = diagonal_attention_scores.new_ones(one_sided_attn_window_size,one_sided_attn_window_size+1).tril().flip(dims=[0])
        beginning_mask = beginning_mask_2d[None,:,None,:]
        ending_mask = beginning_mask.flip(dims=(1,3))
        #到这里的内容一致
        beginning_input = diagonal_attention_scores[:,:one_sided_attn_window_size,:,:one_sided_attn_window_size+1]
        #beginning_mask为beginning_mask_2d的一部分，所以必须先beginning_input从diagonal_attention_scores中提取出内容
        #然后再调用beginning_mask = beginning_mask.expand去扩展diagonal_attention_scores的内容
        beginning_mask = beginning_mask.expand(beginning_input.size())
        
        beginning_input.masked_fill_(beginning_mask == 1,-float("inf"))
        ending_input = diagonal_attention_scores[:,-one_sided_attn_window_size:,:,-(one_sided_attn_window_size+1):]
        ending_mask = ending_mask.expand(ending_input.size())
        ending_input.masked_fill_(ending_mask == 1,-float("inf"))
        return diagonal_attention_scores
    
    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    
    def _sliding_chunks_matmul_attn_probs_value(
        self,attention_probs:torch.Tensor,value:torch.Tensor,one_sided_attn_window_size:int
    ):
        batch_size,seq_len,num_heads,size_per_head = value.size()
        assert seq_len % (one_sided_attn_window_size*2) == 0
        
        assert attention_probs.size()[:3] == value.size()[:3]
        assert attention_probs.size(3) == 2*one_sided_attn_window_size+1
        
        chunked_attention_probs = attention_probs.transpose(1,2).reshape(
            batch_size*num_heads,seq_len//one_sided_attn_window_size,one_sided_attn_window_size,2*one_sided_attn_window_size+1
        )
        
        chunked_attention_probs = nn.functional.pad(
            chunked_attention_probs,(0,one_sided_attn_window_size+1)
        )
        
        chunked_attention_probs = chunked_attention_probs.view(
            batch_size*num_heads,seq_len//one_sided_attn_window_size,one_sided_attn_window_size*(3*one_sided_attn_window_size+2)
        )
        
        chunked_attention_probs = chunked_attention_probs[0:batch_size*num_heads,0:seq_len//one_sided_attn_window_size,0:3*one_sided_attn_window_size*one_sided_attn_window_size+one_sided_attn_window_size]
        
        chunked_attention_probs = chunked_attention_probs.view(batch_size*num_heads,seq_len//one_sided_attn_window_size,one_sided_attn_window_size,3*one_sided_attn_window_size+1)
        
        chunked_attention_probs = chunked_attention_probs[0:batch_size*num_heads,0:seq_len//one_sided_attn_window_size,0:one_sided_attn_window_size,0:3*one_sided_attn_window_size]
        #===================================================
        
        #chunked_attention_probs = chunked_attention_probs[]
        
        value = value.transpose(1,2).reshape(batch_size*num_heads,seq_len,size_per_head)
        #value.size = (2,512,12,64),transpose(1,2)之后value.size = (2,12,512,64)
        #reshape之后value = (24,512,64)
        #new value.size = (24,512,64)
        padded_value = nn.functional.pad(value,(0,0,one_sided_attn_window_size,one_sided_attn_window_size),value=-1)
        #nn.functional.pad四维中四个元素(左填充，右填充，上填充，下填充)，数值代表填充次数
        #padded_value.size = (24,1024,64)
        chunked_value_size = (batch_size*num_heads,seq_len//one_sided_attn_window_size,3*one_sided_attn_window_size,size_per_head)
        chunked_value_stride = padded_value.stride()
        #chunked_value.size = (24,2,768,64),chunked_value_stride = (65536,64,1)
        #(这里的padded_value=(24,1024,64),所以chunked_value_stride=(65536,64,1))
        chunked_value_stride = (
            chunked_value_stride[0],
            one_sided_attn_window_size*chunked_value_stride[1],
            chunked_value_stride[1],
            chunked_value_stride[2],
        )
        #chunked_value_stride = (65536,64*256,64,1) = (65536,16384,64,1)
        chunked_value = padded_value.as_strided(size=chunked_value_size,stride=chunked_value_stride)
        context = torch.matmul(chunked_attention_probs,chunked_value)
        return context.view(batch_size,num_heads,seq_len,size_per_head).transpose(1,2)
    
    def forward(self,inputs,mask=None,is_index_masked=None,**kwargs):
        
        batch_size,seq_len,embedding_size = inputs.size(0),inputs.size(1),inputs.size(2)
        inputs = inputs.transpose(0,1)
        #输入inputs = (2,512,768),transpose之后为(512,2,768)
        query = self.query_layer(inputs)
        key = self.key_layer(inputs)
        value = self.value_layer(inputs)
        assert (
           embedding_size == self.config.embedding_size
        ), f"embedding_size should be {self.config.embedding_size}, but be {embedding_size}"

        query = query/math.sqrt(float(self.config.size_per_head))

        #这里为进入到_sliding_chunks_query_key_matmul函数中的调用内容
        
        assert (
            seq_len%(self.one_sided_attn_window_size*2) == 0
        ), f"Sequence length should be multiple of {self.one_sided_attn_window_size*2}. Given {seq_len}"
        assert query.size() == key.size()
        chunks_count = seq_len//self.one_sided_attn_window_size-1
        
        query = query.view(seq_len,batch_size,self.config.num_attention_heads,self.config.size_per_head).transpose(0,1)
        key = key.view(seq_len,batch_size,self.config.num_attention_heads,self.config.size_per_head).transpose(0,1)
        attention_scores = self._sliding_chunks_query_key_matmul(query,key,self.one_sided_attn_window_size)
        
        remove_from_windowed_attention_mask = (mask !=0)[:,:,None,None]
        float_mask = remove_from_windowed_attention_mask.type_as(query).masked_fill(
            remove_from_windowed_attention_mask,-10000.0
        )
        diagonal_mask = self._sliding_chunks_query_key_matmul(
            float_mask.new_ones(size=float_mask.size()),float_mask,self.one_sided_attn_window_size
        )
        attention_scores += diagonal_mask
        #到这内容完全相同
        assert list(attention_scores.size()) == ([
            batch_size,
            seq_len,
            self.config.num_attention_heads,
            self.one_sided_attn_window_size*2+1
        ]), f"local attention_probs should be of size ({batch_size}, {seq_len}, {self.config.num_attention_heads}, {self.one_sided_attn_window_size*2+1}), but is of size {attention_scores.size()}"
        attention_probs = nn.functional.softmax(
            attention_scores,dim=-1,dtype=torch.float32
        )
        attention_probs = torch.masked_fill(attention_probs,is_index_masked[:,:,None,None],0.0)
        attention_probs = attention_probs.type_as(attention_scores)
        del attention_scores
        attention_probs = nn.functional.dropout(attention_probs,p=self.config.attention_probs_dropout_prob,training=self.training)
        value = value.view(seq_len,batch_size,self.config.num_attention_heads,self.config.size_per_head).transpose(0,1)
        attention_output = self._sliding_chunks_matmul_attn_probs_value(
            attention_probs,value,self.one_sided_attn_window_size
        )
        
        attention_output = attention_output.reshape(batch_size,seq_len,self.config.embedding_size)
        return attention_output
