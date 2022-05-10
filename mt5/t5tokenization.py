import sentencepiece as spm


class T5Tokenizer(object):
    #t5中的inputs和labels千米填充的特殊符号不一样，明天仔细看一下
    def __init__(self,config,vocab_file):
        self.pad_token_id = config.pad_token_id
        self.eos_token_id = config.eos_token_id
        self.decoder_start_token_id = config.decoder_start_token_id
        self.vocab_file = vocab_file
        self.sp_model_kwargs = {}
        self.sp_model = spm.SentencePieceProcessor(self.sp_model_kwargs)
        self.sp_model.Load(vocab_file)
    
    def tokenize(self,text):
        tokenized_text = self.sp_model.encode(text,out_type=str)
        return tokenized_text
    
    def get_input_ids(self,text):
        tokens = self.tokenize(text)
        ids = []
        for token in tokens:
            ids.append(self.sp_model.piece_to_id(token))
        ids.append(self.eos_token_id)
        return ids