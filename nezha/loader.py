import torch
def load_bert_data(model,resolved_archive_file):
    print('!!!load Pytorch model!!!')
    state_dict = None
    if state_dict is None:
        try:
            state_dict = torch.load(resolved_archive_file, map_location="cpu")
            print('state_dict = ')
            print(state_dict.keys())
            print('model state_dict = ')
            print(model.state_dict())
        except Exception:
            raise OSError(
                f"Unable to load weights from pytorch checkpoint file for '{pretrained_model_name_or_path}' "
                f"at '{resolved_archive_file}'"
                "If you tried to load a PyTorch model from a TF 2.0 checkpoint, please set from_tf=True. "
            )
        transformer_dicts = {
           'bertembeddings.word_embeddings_layer.weight':'bert.embeddings.word_embeddings.weight',
           'bertembeddings.position_embeddings_layer.weight':'bert.embeddings.position_embeddings.weight',
           'bertembeddings.segment_embeddings_layer.weight':'bert.embeddings.token_type_embeddings.weight',
           'bertembeddings.layer_normalization.gamma':'bert.embeddings.LayerNorm.gamma',
           'bertembeddings.layer_normalization.beta':'bert.embeddings.LayerNorm.beta',
           'bert_pooler.weight':'bert.pooler.dense.weight',
           'bert_pooler.bias':'bert.pooler.dense.bias'
           r"""
           'bert/mlm_dense0/kernel:0':'cls/predictions/transform/dense/kernel',
           'bert/mlm_dense0/bias:0':'cls/predictions/transform/dense/bias',
           'bert/mlm_dense1/kernel:0':'bert/embeddings/word_embeddings',
           'bert/mlm_dense1/bias:0':'cls/predictions/output_bias',
           'bert/mlm_norm/gamma:0':'cls/predictions/transform/LayerNorm/gamma',
           'bert/mlm_norm/beta:0':'cls/predictions/transform/LayerNorm/beta'
           """
        }
        #由自己的权重名称去找原先的权重名称
        for layer_ndx in range(model.num_layers):
            transformer_dicts.update({
                'bert_encoder_layer.%d.attention.query_layer.weight'%(layer_ndx):'bert/encoder/layer_%d/attention/self/query/kernel'%(layer_ndx),
                #注意中间有冒号，两边要分开进行赋值
                'bert_encoder_layer.%d.attention.query_layer.bias'%(layer_ndx):'bert/encoder/layer_%d/attention/self/query/bias'%(layer_ndx),
                'bert_encoder_layer.%d.attention.key_layer.weight'%(layer_ndx):'bert/encoder/layer_%d/attention/self/key/kernel'%(layer_ndx),
                'bert_encoder_layer.%d.attention.key_layer.bias'%(layer_ndx):'bert/encoder/layer_%d/attention/self/key/bias'%(layer_ndx),
                'bert_encoder_layer.%d.attention.value_layer.weight'%(layer_ndx):'bert/encoder/layer_%d/attention/self/value/kernel'%(layer_ndx),
                'bert_encoder_layer.%d.attention.value_layer.bias'%(layer_ndx):'bert/encoder/layer_%d/attention/self/value/bias'%(layer_ndx),
                
                'bert_encoder_layer.%d.dense0.weight'%(layer_ndx):'bert/encoder/layer_%d/attention/output/dense/kernel'%(layer_ndx),
                'bert_encoder_layer.%d.dense0.bias'%(layer_ndx):'bert/encoder/layer_%d/attention/output/dense/bias'%(layer_ndx),
                'bert_encoder_layer.%d.layer_norm0.weight'%(layer_ndx):'bert/encoder/layer_%d/attention/output/LayerNorm/gamma'%(layer_ndx),
                'bert_encoder_layer.%d.layer_norm0.bias'%(layer_ndx):'bert/encoder/layer_%d/attention/output/LayerNorm/beta'%(layer_ndx),
                
                'bert_encoder_layer.%d.dense.weight'%(layer_ndx):'bert/encoder/layer_%d/intermediate/dense/kernel'%(layer_ndx),
                'bert_encoder_layer.%d.dense.bias'%(layer_ndx):'bert/encoder/layer_%d/intermediate/dense/bias'%(layer_ndx),

                'bert_encoder_layer.%d.dense1.weight'%(layer_ndx):'bert/encoder/layer_%d/output/dense/kernel'%(layer_ndx),
                'bert_encoder_layer.%d.dense1.bias'%(layer_ndx):'bert/encoder/layer_%d/output/dense/bias'%(layer_ndx),
                'bert_encoder_layer.%d.layer_norm1.weight'%(layer_ndx):'bert/encoder/layer_%d/output/LayerNorm/gamma'%(layer_ndx),
                'bert_encoder_layer.%d.layer_norm1.bias'%(layer_ndx):'bert/encoder/layer_%d/output/LayerNorm/beta'%(layer_ndx),
            })
        current_dict = model.state_dict().keys()
        
        print('model.state_dict = ')
        print(model.state_dict().keys())
        print('-------------------')
        print(model.state_dict().values())
        print('@@@@@@@@@@@@@@@@@@@')
        print(model.state_dict()['bert_pooler.weight'])
        model.state_dict()['bert_pooler.weight'] = state_dict[transformer_dicts['bert_pooler.weight']]
        r"""
        model, missing_keys, unexpected_keys, error_msgs = _load_state_dict_into_model(
            model, state_dict, pretrained_model_name_or_path, _fast_init=_fast_init
        )
        """
        return model

def _load_state_dict_into_model(model,state_dict,pretrained_model_name_or_path,_fast_init=True):
    new_state_dict = model.state_dict()
    print('new_state_dict = ')
    print(new_state_dict)
    return None,None,None,None