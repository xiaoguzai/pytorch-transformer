#Author:xiaoguzai
#email:474551240@qq.com
#Download pretrain-file from https://huggingface.co/google/mMT5-base/tree/main
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
r"""
                                                               DecoderLayerAttention
                                       DecoderLayerTransformers
decoder部分的结构图---DecoderTransformers                        
                                                               DecoderLayerAttention
                                       DecoderCrossTransformers
                                                               DecoderCrossAttention
"""
class MT5Config(object):
    def __init__(self,
                 d_ff = 2048,
                 d_kv = 64,
                 d_model = 768,
                 decoder_start_token_id = 0,
                 dropout_rate = 0.1,
                 eos_token_id = 1,
                 feed_forward_proj = 'gated-gelu',
                 initializer_factor = 1.0,
                 is_encoder_decoder = True,
                 layer_norm_epsilon = 1e-06,
                 model_type = 'mMT5',
                 num_decoder_layers = 12,
                 num_heads = 12,
                 num_layers = 12,
                 output_past = True,
                 pad_token_id = 0,
                 relative_attention_num_buckets = 32,
                 tie_word_embeddings = False,
                 tokenizer_class = 'MT5Tokenizer',
                 use_cache = True,
                 vocab_size = 250112,
                *args, **kwargs):
        self.intermediate_size = d_ff
        self.d_ff = d_ff
        
        self.size_per_head = d_kv
        self.embedding_size = d_model

        self.decoder_start_token_id = decoder_start_token_id
        self.dropout_rate = dropout_rate
        self.eos_token_id = eos_token_id
        self.feed_forward_proj = feed_forward_proj
        self.initializer_factor = initializer_factor
        self.is_encoder_decoder = is_encoder_decoder
        self.layer_norm_epsilon = layer_norm_epsilon
        self.model_type = model_type

        self.num_decoder_layers = num_decoder_layers
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.output_past = output_past
        self.pad_token_id = pad_token_id
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.tie_word_embeddings = tie_word_embeddings
        self.use_cache = use_cache
        self.vocab_size = vocab_size

class MT5(nn.Module):
    def __init__(self,config,**kwargs):
        #这里初步先将所有的参数都放入__init__之中
        #后期可以将不需要更改的参数放入build函数之中
        #之前看的内容相当于对params更新name，并且将所有属性中带有name的内容进行更新
        #这样就成功定义了name的对应值
        #super(Nezha, self).__init__()
        super(MT5,self).__init__()
        self.config = config
        self.mt5encoder = MT5Encoder(config)
        self.mt5decoder = MT5Decoder(config,is_first_layer = True)
        #可以在MT5模型之中加入贪婪搜索和**搜索
        
    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            #一般这里的mean=0.0,stddev标准差为1
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    
    def _shift_right(self,input_ids):
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
        shifted_input_ids[..., 0] = decoder_start_token_id
        assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)
        assert torch.all(shifted_input_ids >= 0).item(), "Verify that `shifted_input_ids` has only positive values"
        return shifted_input_ids
    
    def forward(self,input_ids,labels=None):
        if labels != None:
            decoder_ids = self._shift_right(labels)
        else:
            decoder_ids = torch.tensor([[0]])
        print('111decoder_ids = 111')
        print(decoder_ids)
        print('11111111111111111111')
        print('...input_ids = ...')
        print(input_ids)
        print('..................')
        encoderoutput,_ = self.mt5encoder(input_ids)
        #outputs = self.mt5decoder(input_ids=decoder_ids,encoder_output=encoderoutput,\
        #                         past_layer_key_value_list=None,past_cross_key_value_list=None)
        r"""
        encoderoutput = tensor([[[0.3338,0.0653,-0.2473,...]]])
        """
        output_ids,layer_key_value_list,cross_key_value_list = self.mt5decoder(input_ids=decoder_ids,encoder_output=encoderoutput,\
                                                                              past_layer_key_value_list=None,past_cross_key_value_list=None)
        print('###!!!decoder outputs = ###!!!')
        print(output_ids)
        r"""
        第一次调用结束 output_ids = tensor([[2.6664e-01,...]])这里的output_ids对应值一样
        
        layer_key_value_list = 
       [[tensor([[[[2.8158e-01,9.2018e-01,...]]]]),(key_states)(一样)
        tensor([[[[1.2658e+00,-3.6770e-01,...]]]])],(value_states)(一样)
        
        [tensor([[[[1.2744e+00,6.2678e-01,...]]]]),(key_states)(一样)
         tensor([[[[-1.3342e+00,4.3582e-01,...]]]])],(value_states)(一样)
        [tensor([[[[-2.5021e+00,2.8746e+00,...]]]]),(key_states)(一样)
         tensor([[[[-2.1970e-01,-3.0316e-01,...]]]])],(value_states)(一样)
        [tensor([[[[1.9402e-02,-6.6803e-02,...]]]]),(key_states)(一样)
         .....]
       ]
        第一次运行结束之后，这里的output_ids和layer_key_value_list的值与之前的一样
       
        目前这里第一次decoder结束之后的output_ids参数一致，但是中间放入list之中的位置
        参数并不一致
        """
        #print('layer_key_value_list = ')
        #print(layer_key_value_list)
        print('------cross_key_value_list = ------')
        print(cross_key_value_list)
        r"""
        cross_key_value_list = 
       [[tensor([[[[1.2947,-0.8253,-0.3191,...]]]]),
         tensor([[[[2.3500e+00,-1.4729e+00,1.7213e+00,...]]]])],
        [tensor([[[[-2.2410e+00,-3.4751e-01,-1.3781e-01,...]]]]),
         tensor([[[[2.7306e+00,-2.9113e-02,-1.0420e+00,...]]]])],
        [tensor([[[[-7.0765e-01,-2.2788e+00,1.0651e-01,...]]]]),
         tensor([[[[-2.7844e+00,-4.7090e-01,-1.3003e+00,...]]]])],
        [tensor([[[[-2.3379e+00,4.9815e-01,-5.0319e-01,...]]]]),
         tensor([[[[-2.0731e+00,-2.0705e+00,-4.7851e-01,...]]]])],
        [tensor([[[[-2.1173,-2.6076,-3.0661,...]]]]),
         tensor([[[[0.5179,-1.8678,-1.1161,...]]]])],
        [tensor([[[[-9.1990e-01,  1.2521e+00,  7.8553e-01,...]]]]),
         tensor([[[[-1.1053e+00, -1.9523e+00,  6.0366e-01,...]]]])]
        [tensor([[[[2.6066e+00,1.1190e-01,-1.8662e+00,...]]]]),
         tensor([[[[9.3781e-03,6.3560e-01,-2.0190e+00,...]]]])]
        [tensor([[[[-1.4676e+00, -3.7684e+00, -7.3498e-03,...]]]]),
         tensor([[[[0.0839,1.6328,3.5098,...]]]])]
        ......]
        到这里的时候layer_key_value_list与cross_key_value_list的内容一致,以及output_ids内容一致!!!
        """
        #layer_key_value_list和cross_key_value_list分别保留过去六层中的layer_key_value和cross_key_value的值
        print('mt5 output_ids = ')
        print(output_ids)
        print('11111111111111111')
        return output_ids

def gelu_new(x):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class MT5LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Construct a layernorm module in the MT5 style No bias and no subtraction of mean.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        # layer norm should always be calculated in float32
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # convert into float16 if necessary
        if self.weight.dtype == torch.float16:
            hidden_states = hidden_states.to(torch.float16)
        return self.weight * hidden_states
    
class MT5DenseReluDense(nn.Module):
    #MT5-1.0使用的结构
    def __init__(self, config):
        super().__init__()
        self.wi = nn.Linear(config.embedding_size, config.intermediate_size, bias=False)
        self.wo = nn.Linear(config.intermediate_size, config.embedding_size, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states):
        hidden_states = self.wi(hidden_states)
        hidden_states = nn.functional.relu(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.wo(hidden_states)
        return hidden_states


class MT5DenseGatedGeluDense(nn.Module):
    #MT5-1.1使用的结构(即mMT5使用的结构)
    def __init__(self, config):
        super().__init__()
        self.wi_0 = nn.Linear(config.embedding_size, config.intermediate_size, bias=False)
        self.wi_1 = nn.Linear(config.embedding_size, config.intermediate_size, bias=False)
        self.wo = nn.Linear(config.intermediate_size, config.embedding_size, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.gelu_act = gelu_new

    def forward(self, hidden_states):
        hidden_gelu = self.gelu_act(self.wi_0(hidden_states))
        hidden_linear = self.wi_1(hidden_states)
        hidden_states = hidden_gelu * hidden_linear
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.wo(hidden_states)
        return hidden_states
    
class MT5Encoder(nn.Module):
    def __init__(self,
                 config,
                 **kwargs):
        super(MT5Encoder,self).__init__()
        self.config = config
        self.mt5encoderembeddings_layer = nn.Embedding(config.vocab_size,config.embedding_size)
        self.mt5embeddingdropout = nn.Dropout(config.dropout_rate)
        self.mt5encoder_layer = nn.ModuleList()
        for _ in range(config.num_layers):
            MT5encoder_layer_attention = MT5EncoderTransformers(config)
            self.mt5encoder_layer.append(MT5encoder_layer_attention)
        self.final_layer_norm = MT5LayerNorm(config.embedding_size,config.layer_norm_epsilon)
        self.final_dropout = nn.Dropout(config.dropout_rate)
        
    def forward(self,input_ids):
        output = self.mt5encoderembeddings_layer(input_ids)
        output = self.mt5embeddingdropout(output)
        position_bias = None
        past_value_list = None
        for layer_ndx in self.mt5encoder_layer:
            output,position_bias = layer_ndx(output,position_bias)
            print('output1 = ')
            print(output)
            print('..........')
        #到这里都一样
        #!!!易错点：这里有一个final_layer_norm网络层以及一个dropout网络层
        output = self.final_layer_norm(output)
        output = self.final_dropout(output)
        r"""
        output = tensor([[[0.3338,0.0653,-0.2473,...-0.0836],
                          ......]])
        """
        return output,position_bias

class MT5Decoder(nn.Module):
    #is_first_layer = True
    def __init__(self,
                 config,
                 is_first_layer,
                 **kwargs):
        super(MT5Decoder,self).__init__()
        self.mt5decoderembeddings_layer = nn.Embedding(config.vocab_size,config.embedding_size)
        self.mt5embeddingdropout = nn.Dropout(config.dropout_rate)
        #decoder调用embedding层进行操作,这里会放入新的输入,所以需要一个embedding网络层
        self.mt5decoderlayer_transformers_list = nn.ModuleList()
        self.mt5decodercross_transformers_list = nn.ModuleList()
        self.config = config
        self.is_first_layer = is_first_layer
        self.final_layer_norm = MT5LayerNorm(config.embedding_size,config.layer_norm_epsilon)
        self.final_dropout = nn.Dropout(config.dropout_rate)
        past_key_value_list = []
        for index in range(config.num_layers):
            #每次调用的数值遵循同一个原则，第二次调用的时候才更换原则，
            #所以这里面的is_first_layer的值本质上是一样的
            MT5decoderlayer_transformers = MT5DecoderLayerTransformers(config,is_first_layer=(index==0))
            self.mt5decoderlayer_transformers_list.append(MT5decoderlayer_transformers)
            MT5decodercross_transformers = MT5DecoderCrossTransformers(config,is_first_layer=(index==0))
            self.mt5decodercross_transformers_list.append(MT5decodercross_transformers)
        #decoder之中的第一个layerattention并未调用之前encoder的输出结果，
        #到layercrossattention的时候才调用encoder的输出的结果
    def forward(self,input_ids,encoder_output,past_layer_key_value_list=None,past_cross_key_value_list=None):
        #input_ids为decoder的输入，encoder_output为之前encoder部分的输入
        #past_layer_key_value为之前layer_attention输出的key和value的内容
        #past_cross_key_value为之前cross_attention输出的key和value的内容
        #past_layer_position_bias为之前layer_attention输出的key和value的内容
        #past_cross_position_bias为之前cross_attention输出的key和value的内容
        #这里的之前都是上一波预测完之后下一波预测的内容，所谓上一波，就是上一次整个
        #MT5encoder+MT5decoder(第一次)跑下来预测的结果或者MT5decoder跑下来(非第一次)的预测结果
        #print('***input_ids = ***')
        #print(input_ids)
        #print('******************')
        #print('***mt5encoder_output = ***')
        #print(encoder_output)
        #print('**************************')
        r"""
        input_ids = tensor([[0]])
        """
        #!!!is_first_layer问题!!!
        #这里的self.is_first_layer = False,但是不知道为什么后面没用上
        input_ids = self.mt5decoderembeddings_layer(input_ids)
        input_ids = self.mt5embeddingdropout(input_ids)
        #input_ids = tensor([[250099]]),input_ids = tensor([[[2.1719e+00,-3.9375e+00,...]]])
        #第二波的时候关键是下面的循环调用前面的past_key_value和position_bias的部分
        layer_key_value_list,cross_key_value_list = [None]*self.config.num_layers,[None]*self.config.num_layers
        #layer_key_value_list和cross_key_value_list当前这一波模型计算出来的内容
        layer_position_bias,cross_position_bias = None,None
        r"""
        mt5decoder input_ids = 
tensor([[[ 1.7500e+00, -1.6719e+00,  2.4062e+00,  1.8125e+01,  2.8750e+00,
        """
        layer_position_bias = None
        cross_position_bias = None
        for index in range(self.config.num_layers):
            current_decoder_layer_attention = self.mt5decoderlayer_transformers_list[index]
            current_decoder_cross_attention= self.mt5decodercross_transformers_list[index]
            current_decoder_layer_attention.is_first_layer = self.is_first_layer
            current_decoder_cross_attention.is_first_layer = self.is_first_layer
            #这里针对layer_attention和cross_attention中的is_first_layer赋值很关键
            #因为前面的is_first_layer值的改变并不会引起后面的is_first_layer的值改变
            #第一次的时候layer_position_bias和cross_position_bias的值都为None
            #后续的时候layer_position_bias和cross_position_bias接着前面的继续使用
            if self.is_first_layer == True:
                
                input_ids,past_key_value,layer_position_bias = current_decoder_layer_attention(input_ids,encoder_output,layer_key_value_list[index],layer_position_bias)
                layer_key_value_list[index] = past_key_value
                r"""
                is_first_input_ids = tensor([[[tensor([[[-1.6047e+00, -1.0239e+00,  2.2903e+00,  1.3545e+01,  3.6360e+00,
                1.3194e+01, -2.5889e+00, -2.4471e+00,  1.3384e+01,  2.2182e+00,...]]])]]])
                is_first past_key_value = 
                [tensor([[[[ 2.8158e-01,  9.2018e-01,  5.6795e-01,  2.7472e-02,  3.4783e-02,
                ...]]]]),
                tensor([[[[ 1.2658e+00, -3.6770e-01,  3.2545e-01, -8.6813e-02, -7.1645e-02,
                5.6822e-02, -1.3900e+00,...]]]])]
                is_first_layer_position_bias = 
                tensor([[[[3.0312]],[[3.7031]],[[2.4062]],[[3.4219]],[[4.3750]],...]])
                """
                r"""
                print('111input_ids = 111')
                print(input_ids)
                print('111111111111111111')
                print('111past_key_value = 111')
                print(past_key_value)
                print('11111111111111111111111')
                """
                input_ids,past_key_value,cross_position_bias = current_decoder_cross_attention(input_ids,encoder_output,cross_key_value_list[index],cross_position_bias)
                cross_key_value_list[index] = past_key_value
                r"""
                print('222input_ids = 222')
                print(input_ids)
                print('222past_key_value = 222')
                print(past_key_value)
                print('222layer_position_bias = 222')
                print(layer_position_bias)
                print('222cross_position_bias = 222')
                print(cross_position_bias)
                print('2222222222222222222222')
                """
                r"""
                input_ids = 
                tensor([[[ 6.1559e+00,  5.1795e+00, -1.0123e+01,  6.9136e+00,  3.0762e-01,
                2.0537e+01, -4.0555e+00, -3.3963e+01,  6.6520e+00, -8.8820e+00,
                ......]]])
                past_key_value = 
                [tensor([[[[ 1.2947, -0.8253, -0.3191,  ...,  0.8160,  0.8936,  0.9744],
                           [ 0.3110,  0.0431,  0.2989,  ...,  0.0252, -0.9968, -0.7877],
                           .....................]]]),
                 tensor([[[[ 2.3500e+00, -1.4729e+00,  1.7213e+00,  ...,  5.8123e-01, -4.5292e-01,  1.2720e+00],
                           [-5.4824e-02, -1.9664e-01, -9.1395e-01,  ..., -2.3399e-01, -7.9922e-01,  9.6457e-01],
                           .....................]]])]
                layer_position_bias = 
                 tensor([[[[3.0312]],
                          [[3.7031]],
                          ..........
                          ]])
                cross_position_bias = 
                 tensor([[[[0,0,0,0,0]],
                          [[0,0,0,0,0]],
                          .............
                          ]])
                """
                #print('***cross_position_bias = ***')
                #print(cross_position_bias)
                #print('****************************')
                self.is_first_layer = False
            else:
                r"""
                第一次输入的时候，
                decoder_input_ids = 
                tensor([[[ 6.1559e+00,  5.1795e+00, -1.0123e+01,  6.9136e+00,  3.0762e-01,
                           2.0537e+01, -4.0555e+00, -3.3963e+01,  6.6520e+00, -8.8820e+00,
                           .......]]])
                encoder_output = 
                tensor([[[ 0.3338,  0.0653, -0.2473,  ...,  0.1336, -0.1655, -0.0836],
                         [ 0.0510, -0.0056, -0.0015,  ...,  0.0237, -0.0181,  0.0039],
                         [ 0.0621,  0.0078, -0.0176,  ...,  0.0457,  0.0097,  0.0396],
                         [ 0.2604, -0.1341,  0.2746,  ...,  0.2964,  0.4348, -0.2096],
                         [ 0.2767, -0.1186, -0.4137,  ...,  0.1901,  0.2064,  0.2300]]])
                before layer_key_value_list = 
                [tensor([[[[ 2.8158e-01,  9.2018e-01,  5.6795e-01,  2.7472e-02,  3.4783e-02,
                            -5.0866e-01,  7.7950e-01, -1.6907e-01,  2.1843e-01, -1.8774e-01,
                            ............]]]]),
                 tensor([[[[ 1.2658e+00, -3.6770e-01,  3.2545e-01, -8.6813e-02, -7.1645e-02,
                             5.6822e-02, -1.3900e+00,  5.0612e-01, -5.2253e-02,  7.5100e-02,
                            ............]]]])]
                before cross_key_value_list = 
                [tensor([[[[ 1.2947, -0.8253, -0.3191,  ...,  0.8160,  0.8936,  0.9744],
                           [ 0.3110,  0.0431,  0.2989,  ...,  0.0252, -0.9968, -0.7877],
                           .............]]]),
                 tensor([[[[ 2.3500e+00, -1.4729e+00,  1.7213e+00,  ...,  5.8123e-01, -4.5292e-01,  1.2720e+00],
                           [-5.4824e-02, -1.9664e-01, -9.1395e-01,  ..., -2.3399e-01, -7.9922e-01,  9.6457e-01],
                           .............]]])]
                layer_position_bias = None,cross_position_bias = None
                """
                #!!!这里少了一个之前encoder的输出encoder_output(官方的t5_hidden_states作为t5decoder的输入)
                r"""
                print('777input_ids = 777')
                print(input_ids)
                print('777777777777777777')
                print('777cross_position_bias = 777')
                print(cross_position_bias)
                print('7777777777777777777777777777')
                """
                input_ids,past_key_value,layer_position_bias = current_decoder_layer_attention(input_ids,encoder_output,layer_key_value_list[index],layer_position_bias)
                layer_key_value_list[index] = past_key_value
                r"""
                print('999input_ids = 999')
                print(input_ids)
                print('999999999999999999')
                print('cross_position_bias = ')
                print(cross_position_bias)
                print('======================')
                """
                input_ids,past_key_value,cross_position_bias = current_decoder_cross_attention(input_ids,encoder_output,cross_key_value_list[index],cross_position_bias)
                #第一波计算要单独调用的原因:没有之前的layer_key_value_list[index-1]以及cross_key_value_list[index-1]的past_key_value的信息
                cross_key_value_list[index] = past_key_value
                #prin
            r"""
            input_ids = tensor([[[ 1.8940e+06,  3.2698e+05,  4.6616e+05,  1.3140e+05, -5.6306e+05,
            ......]]])
            layer_key_value_list = [[tensor([[[ 2.8158e-01,  9.2018e-01,  5.6795e-01,  2.7472e-02,  3.4783e-02,
            ......]]]),
            tensor([[[ 1.2658e+00, -3.6770e-01,  3.2545e-01, -8.6813e-02, -7.1645e-02,
            ......]]])],
            [tensor([[[-1.2962e-01,  9.1929e-02, -1.2893e-01,  1.2234e+00,  1.3320e+00,
            ......]]]),(不同1)
             tensor([[[-6.9840e-01,  6.5378e-01, -8.8442e-01,  3.2391e-01, -2.5027e-01,
            ......]]])],(不同2)
            [tensor([[[-0.4830,  3.0482, -1.6070,  0.2889, -1.0568, -1.1358,  0.7152,
            ......]]]),
            tensor([[[ 2.1534e-01,  2.8758e+00,  8.0222e-02, -8.4670e-02,  3.2962e+00,
            ......]]]),
            cross_key_value_list = [[tensor([[[  74.8065,   12.6185,  -50.6548,  ...,   23.4672,  -29.2509,
            ......
            """
            r"""
            if index == 2:
                prin
            """
        print('$$!input_ids = $$!')
        print(input_ids)
        print('$$!!!!!!!!!!!!!$$!')
        input_ids = self.final_layer_norm(input_ids)
        input_ids = self.final_dropout(input_ids)
        print('xxxinput_ids = xxx')
        print(input_ids)
        print('xxxxxxxxxxxxxxxxxx')
        print('xxxlayer_key_value_list = xxx')
        print(layer_key_value_list)
        print('xxxcross_key_value_list = xxx')
        print(cross_key_value_list)
        print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
        r"""
        if self.is_first_layer == True:
            print('final input_ids = ')
            print(input_ids)
            print('==================')
            print('layer_key_value_list = ')
            print(layer_key_value_list)
            print('cross_key_value_list = ')
            print(cross_key_value_list)
        在is_first_layer成立的情况下输出下列结果
        layer_key_value_list = 
       [[tensor([[[[2.8158e-01,9.2018e-01,...]]]]),(key_states)(一样)
        tensor([[[[1.2658e+00,-3.6770e-01,...]]]])],(value_states)(一样)
        [tensor([[[[1.2744e+00,6.2678e-01,...]]]]),(key_states)(不一样)
         tensor([[[[-1.3342e+00,4.3582e-01,...]]]])],(value_states)(不一样)
        [tensor([[[[-2.5021e+00,2.8746e+00,...]]]]),(key_states)(不一样)
         tensor([[[[-2.1970e-01,-3.0316e-01,...]]]])],(value_states)(不一样)
        [tensor([[[[1.9402e-02,-6.6803e-02,...]]]]),(key_states)(不一样)
         .....]
       ]
        """
        print('mt5decoder return')
        #经历了第一波之后的is_first_layer的值变为False
        return input_ids,layer_key_value_list,cross_key_value_list

r"""
class DecoderTransformers(nn.Module):
    def __init__(self,
                 config,
                 **kwargs)
"""

class MT5DecoderLayerTransformers(nn.Module):
    def __init__(self,
                 config,
                 is_first_layer,
                 **kwargs):
        super(MT5DecoderLayerTransformers,self).__init__()
        self.decoderlayerattention = MT5DecoderLayerAttention(config,is_first_layer=is_first_layer)
        self.layer_norm0 = MT5LayerNorm(config.embedding_size,eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)
    def forward(self,input_ids,encoder_output,past_key_value,position_bias):
        #past_key_value为一个数组，past_key_value[0] = key_states,past_key_value[1] = value_states
        print('MT5DecoderLayerTransformers forward')
        origin_input_ids = input_ids
        print('origin_input_ids = ')
        print(origin_input_ids)
        print('===================')
        print('encoder_output = ')
        print(encoder_output)
        print('past_key_value = ')
        print(past_key_value)
        print('%%%%%%%%%%%%%%%%%')
        input_ids = self.layer_norm0(input_ids)
        #to this process is the same
        print('normed_hidden_states = ')
        print(input_ids)
        print('=======================')
        r"""
        第一次进入mt5decoderlayertransformers之中
        origin_input_ids = 
        tensor([[[ 1.7500e+00, -1.6719e+00,  2.4062e+00,  1.8125e+01,  2.8750e+00,
                   ...........]]])
        encoder_output = 
tensor([[[ 0.3338,  0.0653, -0.2473,  ...,  0.1336, -0.1655, -0.0836],
         [ 0.0510, -0.0056, -0.0015,  ...,  0.0237, -0.0181,  0.0039],
         [ 0.0621,  0.0078, -0.0176,  ...,  0.0457,  0.0097,  0.0396],
         [ 0.2604, -0.1341,  0.2746,  ...,  0.2964,  0.4348, -0.2096],
         [ 0.2767, -0.1186, -0.4137,  ...,  0.1901,  0.2064,  0.2300]]])
        """
        input_ids,past_key_value,position_bias = self.decoderlayerattention(input_ids,encoder_output,past_key_value,position_bias)
        print('---input_ids = 111')
        print(input_ids)
        print('111origin_output_ids = 111')
        print(origin_input_ids)
        print('11111111111111111111111111')
        
        r"""
        第二次decoderlayerattention进入的时候
        input_ids = 
        tensor([[[-8.7215e-01, -3.4756e-01,  7.9853e+00, -1.1827e+00,  5.3164e+00,
                  ....................]]])
        origin_input_ids = 
        tensor([[[ 6.1559e+00,  5.1795e+00, -1.0123e+01,  6.9136e+00,  3.0762e-01,
                  ....................]]])
        """
        input_ids = origin_input_ids+self.dropout(input_ids)
        r"""
        计算之后input_ids的结果为
        input_ids = 
        tensor([[[ 5.2837e+00,  4.8319e+00, -2.1373e+00,  5.7309e+00,  5.6240e+00,
                   ............................]]])
        
        """
        print('###input_ids = ###')
        print(input_ids)
        print('###past_key_value = ###')
        print(past_key_value)
        print('###position_bias = ###')
        print(position_bias)
        print('######################')
        r"""
        input_ids = 
        tensor([[[5.2837e+00,  4.8319e+00, -2.1373e+00,  5.7309e+00,  5.6240e+00,
                   ...........]]])
        past_key_value = 
       [tensor([[[[ 1.2744e+00,  6.2678e-01, -5.5216e-01, -7.2706e-01, -1.1683e-01,
                   ...........]]]),
        tensor([[[[-1.3342e+00,  4.3582e-01, -4.1409e+00, -1.4343e+00, -2.3472e+00,
                   ...........]]]])]
        position_bias = 
        tensor([[[[  3.0312]],
                 [[  3.7031]],
                 ...........]])
        """
        return input_ids,past_key_value,position_bias

class MT5DecoderCrossTransformers(nn.Module):
    def __init__(self,
                 config,
                 is_first_layer,
                 **kwargs):
        super(MT5DecoderCrossTransformers,self).__init__()
        self.decodercrossattention = MT5DecoderCrossAttention(config,is_first_layer=is_first_layer)
        self.layer_norm0 = MT5LayerNorm(config.embedding_size,eps=config.layer_norm_epsilon)
        self.layer_norm1 = MT5LayerNorm(config.embedding_size,eps=config.layer_norm_epsilon)
        self.mt5densegatedgeludense = MT5DenseGatedGeluDense(config)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.gelu_act = gelu_new
        self.config = config
    def forward(self,input_ids,encoder_output,past_key_value,position_bias):
        origin_input_ids = input_ids
        r"""
        第二次进入mt5decodercrosstransformer的内容
        origin_input_ids = 
        tensor([[[ 5.2837e+00,  4.8319e+00, -2.1373e+00,  5.7309e+00,  5.6240e+00,
                   ..........]]])
        
        """
        print('mt5decodercrossattention input_ids = ')
        print(input_ids)
        print('=====================================')
        print('mt5decodercrossattention encoder_output = ')
        print(encoder_output)
        print('mt5decodercrossattention past_key_value = ')
        print(past_key_value)
        print('position_bias = ')
        print(position_bias)
        print('==========================================')
        r"""
        第一次进入mt5decodercrosstransformers调用的内容
        tensor([[[-1.6047e+00,-1.0239e+00, 2.2903e+00...]]])
        第二次进入mt5decodercrosstransformers调用的内容
        tensor([[[ 5.2837e+00, 4.8319e+00,-2.1373e+00...]]])
        mt5decodercrossattention past_key_value = 
[tensor([[[[ 1.2947, -0.8253, -0.3191,  ...,  0.8160,  0.8936,  0.9744],
        .......]]]),
 tensor([[[[ 2.3500e+00, -1.4729e+00,  1.7213e+00,  ...,  5.8123e-01,
        .......]]]])]
        positin_bias = None
        """
        input_ids = self.layer_norm0(input_ids)
        r"""
        第二次进入mt5decodercrosstransformer的内容
        input_ids = 
        tensor([[[ 2.4437e-02,  1.5726e-02, -7.7189e-03,  1.5952e-02,  1.2925e-02,
                   ..........]]])
        past_key_value = 
        [tensor([[[[ 1.2947, -0.8253, -0.3191,  ...,  0.8160,  0.8936,  0.9744],
         ...................]]]),
         
        """
        print('after layer_norm0')
        print('---input_ids = ---')
        print(input_ids)
        print('------------------')
        print('---past_key_value = ---')
        print(past_key_value)
        print('-----------------------')
        print('---position_bias = ---')
        print(position_bias)
        print('----------------------')
        input_ids,past_key_value,position_bias = self.decodercrossattention(input_ids,encoder_output,past_key_value,position_bias)
        #过了decodercrossattention之后内容不一样，解决多次遇到网络层不好操作的办法：
        #通过某一层中间的值去固定中间的内容
        print('after decodercrossattention')
        print('input_ids = ')
        print(input_ids)
        print('kkkorigin_input_ids = kkk')
        print(origin_input_ids)
        print('===========================')
        r"""
        到这里的值一样
        origin_input_ids = 
        tensor([[[-1.6047e+00, -1.0239e+00,  2.2903e+00,  1.3545e+01,  3.6360e+00,
           1.3194e+01, -2.5889e+00, -2.4471e+00,  1.3384e+01,  2.2182e+00,
           ......]]])
        input_ids = 
        tensor([[[ 3.3999e+00,  1.0152e+00, -8.7780e-01,  2.2120e+00, -1.1477e-01,
           7.1591e-01, -5.7654e-01, -3.9603e+00,  2.4773e+00, -1.2417e+00,
           ......]]])
        """
        #input_ids = origin_input_ids+self.dropout(input_ids[0])
        input_ids = origin_input_ids+self.dropout(input_ids)
        r"""
        input_ids = 
        tensor([[[ 1.7952e+00, -8.7521e-03,  1.4125e+00,  1.5757e+01,  3.5213e+00,
           1.3910e+01, -3.1655e+00, -6.4074e+00,  1.5861e+01,  9.7646e-01,
           ......]]])
        """
        if torch.isinf(input_ids).any():
            clamp_value = torch.finfo(input_ids.dtype).max - 1000
            input_ids = torch.clamp(input_ids, min=-clamp_value, max=clamp_value)
        origin_input_ids = input_ids
        print('&&&origin_input_ids = &&&')
        print(origin_input_ids)
        print('&&&&&&&&&&&&&&&&&&&&&&&&&')
        r"""
        origin_input_ids = 
        tensor([[[ 1.7952e+00, -8.7521e-03,  1.4125e+00,  1.5757e+01,  3.5213e+00,
                   1.3910e+01, -3.1655e+00, -6.4074e+00,  1.5861e+01,  9.7646e-01,
                   ..............]]])
        
        """
        input_ids = self.layer_norm1(input_ids)
        input_ids = self.mt5densegatedgeludense(input_ids)
        input_ids = origin_input_ids+self.dropout(input_ids)
        print('after MT5DecoderCrossTransformers')
        print('```input_ids = ```')
        print(input_ids)
        print('``````````````````')
        print('```past_key_value = ```')
        print(past_key_value)
        print('```````````````````````')
        print('```position_bias = ```')
        print(position_bias)
        print('``````````````````````')
        r"""
        第一次这里输出的时候
        input_ids = 
tensor([[[ 6.1559e+00,  5.1795e+00, -1.0123e+01,  6.9136e+00 ...]]])
        past_key_value = 
[tensor([[[[ 1.2947, -0.8253, -0.3191, ...]]]]),
 tensor([[[[ 2.3500e+00, -1.4729e+00, ...]]]])]
        position_bias = 
  tenosr([[[[0,0,0,0,0]],
           [[0,0,0,0,0]],
           ...........]])
        这里的position_bias参数没有能够对上
        第二次decodercrossattention的时候
        input_ids = 
tensor([[[ 1.0544e+01,  1.2843e+00, -1.0977e+00, -9.2583e+00,  5.0282e-01,
           6.2444e+00, -6.3078e+00,  1.3156e+01,  1.4648e+01,  1.2124e+01,
           ..........]]])
         """
        return input_ids,past_key_value,position_bias

class MT5EncoderTransformers(nn.Module):
#这里目前选用的是MT5-1.1的版本，效果更好
    def __init__(self,
                 config,
                 **kwargs):
        super(MT5EncoderTransformers,self).__init__()
        self.mt5encoderlayerattention = MT5EncoderLayerAttention(config)
        self.mt5layernorm0 = MT5LayerNorm(config.embedding_size,config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)
        #这里注意使用nn.Dropout,F.Dropout之中有坑
        self.mt5densegatedgeludense = MT5DenseGatedGeluDense(config)
        self.mt5layernorm1 = MT5LayerNorm(config.embedding_size,config.layer_norm_epsilon)
		
    def forward(self,input_ids,position_bias):
        origin_input_ids = input_ids
        input_ids = self.mt5layernorm0(input_ids)
        input_ids,position_bias = self.mt5encoderlayerattention(input_ids,position_bias)
        input_ids = origin_input_ids+self.dropout(input_ids)
        #到这里也是一样的
        origin_input_ids = input_ids
        input_ids = self.mt5layernorm1(input_ids)
        input_ids = self.mt5densegatedgeludense(input_ids)
        input_ids = origin_input_ids+self.dropout(input_ids)
        #还是这里的self.dropout出问题!!!
        return input_ids,position_bias

class MT5EncoderLayerAttention(nn.Module):
    def __init__(self,
                 config,
                 **kwargs):
        super(MT5EncoderLayerAttention,self).__init__()
        self.query_layer = nn.Linear(config.embedding_size,config.num_heads*config.size_per_head,bias=False)
        #这里MT5-base的时候config.num_heads*config.size_per_head == config.embedding_size,
        #MT5-small的时候两者并不相等
        self.key_layer = nn.Linear(config.embedding_size,config.num_heads*config.size_per_head,bias=False)
        self.value_layer = nn.Linear(config.embedding_size,config.num_heads*config.size_per_head,bias=False)
        self.output_layer = nn.Linear(config.num_heads*config.size_per_head,config.embedding_size,bias=False)
        self.relative_attention_bias = nn.Embedding(config.relative_attention_num_buckets,config.num_heads)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.config = config
        
    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            #一般这里的mean=0.0,stddev标准差为1
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    
    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

        Translate relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the model has been trained on
        (我们对于小的相对绝对位置使用小的bucket，对于大的相对绝对位置使用大的buckets，所有相对位置超出max_distance都映射到相同的bucket
        之中，所有相对位置小于等于max_distance都映射到相同的bucket之中，这比模型训练允许更多更优雅的更长序列的归一化操作)
        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer

        Returns:
            a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
        """
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        # now relative_position is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        relative_postion_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_postion_if_large = torch.min(
            relative_postion_if_large, torch.full_like(relative_postion_if_large, num_buckets - 1)
        )

        relative_buckets += torch.where(is_small, relative_position, relative_postion_if_large)
        return relative_buckets

    def compute_bias(self, query_length, key_length):
        """Compute binned relative position bias"""
        context_position = torch.arange(
            query_length, dtype=torch.long, device=self.relative_attention_bias.weight.device
        )[:, None]
        memory_position = torch.arange(
            key_length, dtype=torch.long, device=self.relative_attention_bias.weight.device
        )[None, :]
        relative_position = memory_position - context_position  # shape (query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            bidirectional=True,
            num_buckets=self.config.relative_attention_num_buckets,
        )
        values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        #这里的relative_attention_bias调用nn.embedding的模型内容
        values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        return values

    
    def forward(self,input_ids,position_bias=None):
        #position_bias通过传递的方式，减少模型的计算过程，加快模型的运算速度
        batch_size,seq_length = input_ids.shape[:2]
        query = self.query_layer(input_ids)
        key = self.key_layer(input_ids)
        value = self.value_layer(input_ids)
        query = query.view(batch_size,-1,self.config.num_heads,self.config.size_per_head).transpose(1,2)
        #-1的意思是这个位置让电脑帮我们计算具体的维度内容
        key = key.view(batch_size,-1,self.config.num_heads,self.config.size_per_head).transpose(1,2)
        value = value.view(batch_size,-1,self.config.num_heads,self.config.size_per_head).transpose(1,2)
        scores = torch.matmul(
            query,key.transpose(3,2)
        )
        real_seq_length,key_length = input_ids.shape[1],input_ids.shape[1]
        if position_bias == None:
            position_bias = self.compute_bias(real_seq_length,key_length)
        scores += position_bias
        #到这里目前内容一致
        attn_weights = F.softmax(scores,dim=-1)
        #!!!这里的softmax一定要注明dim=-1
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights,value)
        attn_output = attn_output.transpose(1,2).contiguous().view(batch_size,-1,self.config.num_heads*self.config.size_per_head)
        attn_output = self.output_layer(attn_output)
        #the same
        return attn_output,position_bias

class MT5DecoderLayerAttention(nn.Module):
    def __init__(self,
                 config,
                 is_first_layer,
                 **kwargs):
        super(MT5DecoderLayerAttention,self).__init__()
        self.query_layer = nn.Linear(config.embedding_size,config.num_heads*config.size_per_head,bias=False)
        #这里MT5-base的时候config.num_heads*config.size_per_head == config.embedding_size,
        #MT5-small的时候两者并不相等
        self.key_layer = nn.Linear(config.embedding_size,config.num_heads*config.size_per_head,bias=False)
        self.value_layer = nn.Linear(config.embedding_size,config.num_heads*config.size_per_head,bias=False)
        self.output_layer = nn.Linear(config.num_heads*config.size_per_head,config.embedding_size,bias=False)
        self.relative_attention_bias = nn.Embedding(config.relative_attention_num_buckets,config.num_heads)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.config = config
        self.is_first_layer = is_first_layer
        
    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            #一般这里的mean=0.0,stddev标准差为1
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    
    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=False, num_buckets=32, max_distance=128):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

        Translate relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the model has been trained on
        (我们对于小的相对绝对位置使用小的bucket，对于大的相对绝对位置使用大的buckets，所有相对位置超出max_distance都映射到相同的bucket
        之中，所有相对位置小于等于max_distance都映射到相同的bucket之中，这比模型训练允许更多更优雅的更长序列的归一化操作)
        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer

        Returns:
            a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
        """
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        # now relative_position is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        relative_postion_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_postion_if_large = torch.min(
            relative_postion_if_large, torch.full_like(relative_postion_if_large, num_buckets - 1)
        )

        relative_buckets += torch.where(is_small, relative_position, relative_postion_if_large)
        return relative_buckets

    def compute_bias(self, query_length, key_length):
        """Compute binned relative position bias"""
        context_position = torch.arange(
            query_length, dtype=torch.long, device=self.relative_attention_bias.weight.device
        )[:, None]
        memory_position = torch.arange(
            key_length, dtype=torch.long, device=self.relative_attention_bias.weight.device
        )[None, :]
        relative_position = memory_position - context_position  # shape (query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            bidirectional=False,
            num_buckets=self.config.relative_attention_num_buckets,
        )
        values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        #这里的relative_attention_bias调用nn.embedding的模型内容
        values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        return values

    
    def forward(self,input_ids,encoder_output,past_key_value=None,position_bias=None):
        #position_bias通过传递的方式，减少模型的计算过程，加快模型的运算速度
        batch_size,seq_length = input_ids.shape[:2]
        if self.is_first_layer == True:
            #print('Decoder LayerAttention is_first_layer == True')
            #第一次调用的时候，input_ids使用的是同一部分的内容
            query = self.query_layer(input_ids)
            key = self.key_layer(input_ids)
            value = self.value_layer(input_ids)
        else:
            #第二次调用的时候，input_ids调用的是之前的key和value的内容
            r"""
            print('Decoder LayerAttention is_first_layer == False')
            print('###input_ids = ###')
            print(input_ids)
            """
            query = self.query_layer(input_ids)
            r"""
            print('first query_position')
            print('...query = ...')
            print(query)
            print('..............')
            """
            #key = torch.cat([past_key_value[0],input_ids],dim=2)
            key = self.key_layer(input_ids)
            #past_key_value[0]保存上一波的key的值
            #value = torch.cat([past_key_value[1],input_ids],dim=2)
            #key = self.key_layer(key)
            #value = self.value_layer(value)
            value = self.value_layer(input_ids)
            r"""
            print('```key = ```')
            print(key)
            print('```value = ```')
            print(value)
            """
            #value的值目前不一样---past_
        
        query = query.view(batch_size,-1,self.config.num_heads,self.config.size_per_head).transpose(1,2)
        #-1的意思是这个位置让电脑帮我们计算具体的维度内容
        key = key.view(batch_size,-1,self.config.num_heads,self.config.size_per_head).transpose(1,2)
        value = value.view(batch_size,-1,self.config.num_heads,self.config.size_per_head).transpose(1,2)
        r"""
        print('111key = 111')
        print(key)
        print('111111111111')
        print('111value = 111')
        print(value)
        print('11111111111111')
        """
        scores = torch.matmul(
            query,key.transpose(3,2)
        )
        past_key_value = [key,value]
        #!!!注意这里的key所在的位置
        real_seq_length,key_length = input_ids.shape[1],input_ids.shape[1]
        if position_bias == None:
            #在decoderlayerselfattention中使用的是计算bias的函数内容
            position_bias = self.compute_bias(real_seq_length,key_length)
        scores += position_bias
        #到这里scores的内容都一样
        r"""
        scores = 
tensor([[[[  2.4754]],
         [[  6.1037]],
         [[ -2.4682]],
         [[  2.2286]],
         [[ -1.4690]],
         [[ -0.3556]],
         [[  2.7988]],
         [[-25.5179]],
         [[  3.3651]],
         [[ -0.4366]],
         [[  8.4352]],
         [[ -2.4953]]]], grad_fn=<AddBackward0>)
        """
        attn_weights = F.softmax(scores,dim=-1)
        #这里必须加上dim=-1,默认的dim应该等于1
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights,value)
        attn_output = attn_output.transpose(1,2).contiguous().view(batch_size,-1,self.config.num_heads*self.config.size_per_head)
        attn_output = self.output_layer(attn_output)
        r"""
        print('$$$attn_output = $$$')
        print(attn_output)
        print('$$$$$$$$$$$$$$$$$$$$')
        print('$$$past_key_value = $$$')
        print(past_key_value)
        print('$$$$$$$$$$$$$$$$$$$$$$$')
        print('$$$position_bias = $$$')
        print(position_bias)
        print('$$$$$$$$$$$$$$$$$$$$$$')
        """
        return attn_output,past_key_value,position_bias

class MT5DecoderCrossAttention(nn.Module):
    def __init__(self,
                 config,
                 is_first_layer,
                 **kwargs):
        super(MT5DecoderCrossAttention,self).__init__()
        self.query_layer = nn.Linear(config.embedding_size,config.num_heads*config.size_per_head,bias=False)
        #这里MT5-base的时候config.num_heads*config.size_per_head == config.embedding_size,
        #MT5-small的时候两者并不相等
        self.key_layer = nn.Linear(config.embedding_size,config.num_heads*config.size_per_head,bias=False)
        self.value_layer = nn.Linear(config.embedding_size,config.num_heads*config.size_per_head,bias=False)
        self.output_layer = nn.Linear(config.num_heads*config.size_per_head,config.embedding_size,bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.config = config
        self.is_first_layer = is_first_layer
        
    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            #一般这里的mean=0.0,stddev标准差为1
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    
    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=False, num_buckets=32, max_distance=128):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

        Translate relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the model has been trained on
        (我们对于小的相对绝对位置使用小的bucket，对于大的相对绝对位置使用大的buckets，所有相对位置超出max_distance都映射到相同的bucket
        之中，所有相对位置小于等于max_distance都映射到相同的bucket之中，这比模型训练允许更多更优雅的更长序列的归一化操作)
        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer

        Returns:
            a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
        """
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        # now relative_position is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        relative_postion_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_postion_if_large = torch.min(
            relative_postion_if_large, torch.full_like(relative_postion_if_large, num_buckets - 1)
        )

        relative_buckets += torch.where(is_small, relative_position, relative_postion_if_large)
        return relative_buckets

    def compute_bias(self, query_length, key_length):
        """Compute binned relative position bias"""
        context_position = torch.arange(
            query_length, dtype=torch.long, device=self.query_layer.weight.device
        )[:, None]
        memory_position = torch.arange(
            key_length, dtype=torch.long, device=self.query_layer.weight.device
        )[None, :]
        relative_position = memory_position - context_position  # shape (query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            bidirectional=False,
            num_buckets=self.config.relative_attention_num_buckets,
        )
        values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        #这里的relative_attention_bias调用nn.embedding的模型内容
        values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        return values

    
    def forward(self,input_ids,encoder_output,past_key_value=None,position_bias=None):
        #position_bias通过传递的方式，减少模型的计算过程，加快模型的运算速度
        r"""
        print('mt5crossattention forward')
        print('333input_ids = 333')
        print(input_ids)
        print('333333333333333333')
        """
        r"""
        第一次进入t5crossattention网络层之中
        input_ids = 
        tensor([[[-4.2509e-02, -2.3015e-02,  5.6994e-02,  2.3650e-01,  8.4646e-02,
                  ...........]]])
        
        """
        batch_size,seq_length = input_ids.shape[:2]
        if self.is_first_layer == True:
            r"""
            print('crossattention is_first_layer')
            print('input_ids = ')
            print(input_ids)
            print('============')
            """
            query = self.query_layer(input_ids)
            #print('query1 = ')
            #print(query)
            key = self.key_layer(encoder_output)
            #print('key1 = ')
            #print(key)
            value = self.value_layer(encoder_output)
            #print('value1 = ')
            #print(value)
            #print('111111111')
        else:
            r"""
            print('%%%mt5crossattention forward%%%')
            print('%%%input_ids = %%%')
            print(input_ids)
            print('%%%input_ids.size = %%%')
            print(input_ids.size())
            #input_ids.size() = (1,1,768)
            print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
            print('%%%past_key_value = %%%')
            print(past_key_value)
            print('%%%%%%%%%%%%%%%%%%%%%%%')
            print('%%%position_bias = %%%')
            print(position_bias)
            print('%%%%%%%%%%%%%%%%%%%%%%')
            """
            r"""
            第二次进入mt5crossattention网络层之后
            normed_hidden_states = 
tensor([[[ 2.4437e-02,  1.5726e-02, -7.7189e-03,  1.5952e-02,  1.2925e-02,
           7.8303e-02, -2.0555e-02, -7.4466e-02,  1.8700e-02, -3.2498e-02,
           ..........)
            past_key_value = 
      [
tensor([[[[ 1.2947, -0.8253, -0.3191,  ...,  0.8160,  0.8936,  0.9744,
           ..........]]])
tensor([[[[ 2.3500e+00, -1.4729e+00,  1.7213e+00,  ...,  5.8123e-01,
           ..........]]])
      ]
          这里的past_key_value官方给出的值为None
          position_bias = 
tensor([[[[0., 0., 0., 0., 0.]],
         [[0., 0., 0., 0., 0.]],
            .............]])
            """
            query = self.query_layer(input_ids)
            #这里input_ids与past_key_value[0]连接之前需要变换维度
            key = self.key_layer(encoder_output)
            #past_key_value[0]保存上一波的key的值
            value = self.value_layer(encoder_output)
            r"""
            print('~~~query = ~~~')
            print(query)
            print('~~~key = ~~~')
            print(key)
            print('~~~value = ~~~')
            print(value)
            print('~~~~~~~~~~~~~~')
            print('111query.size = 111')
            print(query.size())
            """
        query = query.view(batch_size,-1,self.config.num_heads,self.config.size_per_head).transpose(1,2)
        #-1的意思是这个位置让电脑帮我们计算具体的维度内容
        key = key.view(batch_size,-1,self.config.num_heads,self.config.size_per_head).transpose(1,2)
        value = value.view(batch_size,-1,self.config.num_heads,self.config.size_per_head).transpose(1,2)
        scores = torch.matmul(
            query,key.transpose(3,2)
        )
        r"""
        print('```scores = ```')
        print(scores)
        print('```````````````')
        """
        if position_bias is None:
            #在decodercrossattention中使用的为全零的矩阵
            encoder_length = encoder_output.shape[1]
            position_bias = torch.zeros(1,self.config.num_heads,seq_length,encoder_length)
        past_key_value = [key,value]
        real_seq_length,key_length = input_ids.shape[1],input_ids.shape[1]
        #if position_bias == None:
        #position_bias = self.compute_bias(real_seq_length,key_length)
        #if self.first_layer == False:
        #    position_bias = position_bias[:,:,-position_bias.size(1):,:]
        r"""
        print('```position_bias = ```')
        print(position_bias)
        print('``````````````````````')
        """
        scores += position_bias
        r"""
        print('111scores = 111')
        print(scores)
        print('111111111111111')
        """
        attn_weights = F.softmax(scores,dim=-1)
        attn_weights = self.dropout(attn_weights)
        r"""
        print('***attn_weights = ***')
        print(attn_weights)
        print('*********************')
        """
        attn_output = torch.matmul(attn_weights,value)
        attn_output = attn_output.transpose(1,2).contiguous().view(batch_size,-1,self.config.num_heads*self.config.size_per_head)
        attn_output = self.output_layer(attn_output)
        r"""
        第一次t5decodercrossattention的输出内容
        attn_output = 
tensor([[[ 3.3999e+00,  1.0152e+00, -8.7780e-01,  2.2120e+00, -1.1477e-01,
        .......]]])
        past_key_value = 
[tensor([[[[ 1.2947, -0.8253, -0.3191,...]]]]),
 tensor([[[[ 2.3500e+00, -1.4729e+00,  1.7213e+00,  ...]]]])]
        position_bias = 
 tensor([[[[0,0,0,0,0]],
          [[0,0,0,0,0]],
          .............]])
        第二次t5decodercrossattention的输出内容
        attn_output = 
tensor([[[ 2.4696e+00,  2.4129e+00, -4.4677e+00, -6.6841e-01, -1.0882e+01,
        ........]]]),
        past_key_value = 
[tensor([[[[-2.2410e+00, -3.4751e-01, -1.3781e-01,  ..., -1.6481e+00,
        ........]]]]),
 tensor([[[[ 2.7306e+00, -2.9113e-02, -1.0420e+00,  ...,  1.7375e+00,
        ........]]]])]
        position_bias = 
tensor([[[[0., 0., 0., 0., 0.]],
         [[0., 0., 0., 0., 0.]],
         [[0., 0., 0., 0., 0.]],
         ....................]])
        """
        r"""
        print('second t5decodercrossattention')
        print('111attn_output = 111')
        print(attn_output)
        print('111past_key_value = 111')
        print(past_key_value)
        print('111position_bias = 111')
        print(position_bias)
        print('1111111111111111111111')
        """
        return attn_output,past_key_value,position_bias
	