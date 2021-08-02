import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import math

class Bert(nn.Module):
    def __init__(self,
                initializer_range=0.02,#1
                embedding_size=768,#4
                project_embeddings_with_bias=True,#5
                vocab_size=21128,#6
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
                pooler_num_attention_heads = 12,
                pooler_num_fc_layers = 3,
                pooler_size_per_head = 128,
                pooler_type = "first_token_transform",
                type_vocab_size = 2,
                with_mlm = False,
                mlm_activation = 'softmax',
                mode = 'bert',
                #max_relative_position = 512,
                solution = 'seq2seq',
                max_relative_position = 64,
                with_pooler = True,
                max_position_embeddings = 512,
                *args, **kwargs):
        #这里初步先将所有的参数都放入__init__之中
        #后期可以将不需要更改的参数放入build函数之中
        #之前看的内容相当于对params更新name，并且将所有属性中带有name的内容进行更新
        #这样就成功定义了name的对应值
        #super(Nezha, self).__init__()
        super(Bert,self).__init__()
        print('__init__ Nezha')
        self.name = 'bert'
        self.initializer_range = initializer_range#1
        self.embedding_size = embedding_size#4
        self.project_embeddings_with_bias = project_embeddings_with_bias#5
        self.vocab_size = vocab_size#6
        self.token_type_vocab_size = 2#9
        self.hidden_dropout = hidden_dropout#10
        self.extra_tokens_vocab_size = extra_tokens_vocab_size#11
        self.project_position_embeddings = project_position_embeddings#12
        self.mask_zero = mask_zero#13
        self.adapter_size = adapter_size#14
        self.adapter_init_scale = adapter_init_scale#16
        self.num_attention_heads = num_attention_heads#17注意力头数，需指定
        assert embedding_size%num_attention_heads == 0,"size_per_head必须能够整除num_attention_heads"
        self.size_per_head = embedding_size//num_attention_heads#18
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
        self.with_mlm = with_mlm
        self.mlm_activation = mlm_activation
        self.mode = mode
        self.solution = solution
        self.max_position_embeddings = max_position_embeddings
        self.max_relative_position = max_relative_position
        self.with_pooler = with_pooler
        self.bertembeddings = Embeddings(vocab_size = self.vocab_size,
                          embedding_size = self.embedding_size,
                          mask_zero = self.mask_zero,
                          max_position_embeddings = self.max_position_embeddings,
                          token_type_vocab_size = self.token_type_vocab_size,
                          hidden_dropout = self.hidden_dropout)
        self.bert_encoder_layer = nn.ModuleList()
        for layer_ndx in range(self.num_layers):
            encoder_layer = Transformer(initializer_range = self.initializer_range,
                                           num_attention_heads = self.num_attention_heads,
                                           embedding_size = self.embedding_size,
                                           size_per_head = self.size_per_head,
                                           attention_probs_dropout_prob = 0.1,
                                           negative_infinity = -10000.0,
                                           intermediate_size = self.intermediate_size,
                                           mode = self.mode,
                                           solution = self.solution,
                                           max_relative_position = self.max_relative_position
                                          )
            self.bert_encoder_layer.append(encoder_layer)
        if self.with_pooler:
            self.bert_pooler = nn.Linear(embedding_size,embedding_size)
    
    def forward(self,inputs):
        outputs = self.bertembeddings(inputs)
        print('embedding_outputs = ')
        print(outputs)
        for layer_ndx in self.bert_encoder_layer:
            outputs = layer_ndx(outputs)
        if self.with_pooler:
            outputs = self.bert_pooler(outputs)
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
    def __init__(self,
                 vocab_size = 30522,
                 embedding_size = 768,
                 mask_zero = False,
                 max_position_embeddings = 512,
                 token_type_vocab_size = 2,
                 initializer_range = 0.02,
                 hidden_dropout = 0.1):
        #之前__init__之中少写了一个self,报错multiple initialize
        #先能跑通一个网络层，再往里面加入网络层
        super(Embeddings, self).__init__()
        self.word_embeddings_layer = nn.Embedding(vocab_size,embedding_size)
        self.segment_embeddings_layer = nn.Embedding(token_type_vocab_size,embedding_size)
        self.position_embeddings_layer = nn.Embedding(max_position_embeddings,embedding_size)
        self.layer_normalization = nn.LayerNorm(embedding_size,eps=1e-12)
        self.dropout_layer = nn.Dropout(hidden_dropout)

    def forward(self,inputs):
        if inputs.size(0) == 2:
            input_ids,segment_ids = inputs[0],inputs[1]
            mask_ids = None
        elif inputs.size(0) == 3:
            input_ids,segment_ids,mask_ids = inputs[0],inputs[1],inputs[2]
        else:
            input_ids = inputs
            segment_ids = torch.zeros_like(inputs)
            mask_ids = None
        seq_len = input_ids.size(1)
        #长度为(batch_size,seq_len)
        position_ids = torch.arange(seq_len,dtype=torch.long,device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        #pos.unsqueeze(0)由(seq_len,)得到(1,seq_len)，
        #接下来使用expand_as(x)由(1,seq_len)->(batch_size,seq_len)
        results = self.word_embeddings_layer(input_ids)+self.segment_embeddings_layer(segment_ids)+self.position_embeddings_layer(position_ids)
        results = self.layer_normalization(results)
        results = self.dropout_layer(results)
        return results

def get_activation(activation):
    if activation == 'gelu':
        return F.gelu
    elif activation == 'relu':
        return F.relu
    elif activation == 'tanh':
        return F.tanh

class Transformer(nn.Module):
    def __init__(self,
                 initializer_range = 0.02,
                 embedding_size = 768,
                 hidden_dropout = 0.1,
                 adapter_size = None,
                 hidden_act = 'gelu',
                 adapter_init_scale = 0.001,
                 num_attention_heads = 12,
                 size_per_head = 64,
                 attention_probs_dropout_prob = 0.1,
                 negative_infinity = -10000.0,
                 intermediate_size = 3072,
                 mode = 'bert',
                 solution = 'seq2seq',
                 max_relative_position = 64,
                 **kwargs):
        super(Transformer,self).__init__()
        self.initializer_range = initializer_range
        self.embedding_size = embedding_size
        self.hidden_dropout = hidden_dropout
        self.adapter_size = adapter_size
        self.hidden_act = hidden_act
        self.adapter_init_scale = adapter_init_scale
        self.num_attention_heads = num_attention_heads
        self.size_per_head = size_per_head
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.negative_infinity = negative_infinity
        self.intermediate_size = intermediate_size
        self.mode = mode
        self.solution = solution
        self.max_relative_position = max_relative_position
        self.attention = AttentionLayer(initializer_range = self.initializer_range,
                                        num_attention_heads = self.num_attention_heads,
                                        size_per_head = self.size_per_head,
                                        attention_probs_dropout_prob = self.attention_probs_dropout_prob,
                                        negative_infinity = self.negative_infinity,
                                        #name = "attention",
                                        mode = self.mode,
                                        solution = self.solution,
                                        max_relative_position = self.max_relative_position)
        self.dense0 = nn.Linear(embedding_size,embedding_size)
        self.dropout0 = nn.Dropout(attention_probs_dropout_prob)
        self.layer_norm0 = nn.LayerNorm(embedding_size,eps=1e-12)
        self.dense = nn.Linear(embedding_size,intermediate_size)
        self.activation = get_activation(hidden_act)
        self.dense1 = nn.Linear(intermediate_size,embedding_size)
        self.dropout1 = nn.Dropout(attention_probs_dropout_prob)
        self.layer_norm1 = nn.LayerNorm(embedding_size,eps=1e-12)
        
    
    def forward(self,inputs,mask=None,**kwargs):
        residual = inputs
        embedding_output = self.attention(inputs)
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

class AttentionLayer(nn.Module):
    def __init__(self,
                 initializer_range = 0.02,
                 num_attention_heads = 12,
                 size_per_head = 64,
                 query_activation = None,
                 key_activation = None,
                 value_activation = None,
                 attention_probs_dropout_prob = 0.1,
                 negative_infinity = -10000.0,
                 mode = 'bert',
                 solution = 'seq2seq',
                 max_relative_position = 64,
                 maxlen = 128,
                 **kwargs):
        super(AttentionLayer,self).__init__()
        self.initializer_range = initializer_range
        self.num_attention_heads = num_attention_heads
        self.size_per_head = size_per_head
        self.query_activation = query_activation
        self.key_activation = key_activation
        self.value_activation = value_activation
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.negative_infinity = negative_infinity
        self.query_layer = None
        self.key_layer = None
        self.value_layer = None
        
        self.supports_masking = True
        self.initializer_range = 0.02
        self.mode = mode
        self.solution = solution
        self.maxlen = maxlen
        self.max_relative_position = max_relative_position
        embedding_size = num_attention_heads*size_per_head
        self.query_layer = nn.Linear(embedding_size,embedding_size)
        self.key_layer = nn.Linear(embedding_size,embedding_size)
        self.value_layer = nn.Linear(embedding_size,embedding_size)
        
    
    def forward(self,inputs,mask=None,**kwargs):
        query = self.query_layer(inputs)
        key = self.key_layer(inputs)
        value = self.value_layer(inputs)

        batch_size,seq_len,embedding_size = inputs.size(0),inputs.size(1),inputs.size(2)
        query = query.view([batch_size,seq_len,self.num_attention_heads,self.size_per_head])
        #query = (1,5,12,64)
        query = query.permute(0,2,1,3)
        key = key.view([batch_size,seq_len,self.num_attention_heads,self.size_per_head])
        key = key.permute(0,2,1,3)
        attention_scores = torch.matmul(query,key.transpose(-1,-2))
        #attention_scores = [1,12,5,64]*[1,12,64,5] = [1,12,5,5]
        if mask is not None:
            mask = mask[:, None, None, :].float()
            scores -= 10000.0 * (1.0 - mask)
        attention_scores = attention_scores/math.sqrt(float(self.size_per_head))
        r"""
        if self.mode == 'unilm':
            if self.solution == 'seq2seq':
                bias_data = self.seq2seq_compute_attention_bias(segment_ids)
                #当批次为128的时候，bias_data = (1,128,128)
                #attention_scores = attention_scores+bias_data[:,None,:,:]
                attention_scores = attention_scores+bias_data[:,None,:,:]
                #(5,12,128,128)+(5,128,128) = (5,12,128,128)
            elif self.solution == 'lefttoright':
                bias_data = self.lefttoright_compute_attention_bias(segment_ids)
                attention_scores = attention_scores+bias_data
         """
        attention_scores = F.softmax(attention_scores,dim=-1)
        value = value.view(batch_size,seq_len,self.num_attention_heads,self.size_per_head)
        value = value.permute(0,2,1,3)
        context_layer = torch.matmul(attention_scores,value)
        context_layer = context_layer.permute(0,2,1,3)
        context_layer = context_layer.contiguous().view(batch_size,seq_len,self.num_attention_heads*self.size_per_head)
        return context_layer