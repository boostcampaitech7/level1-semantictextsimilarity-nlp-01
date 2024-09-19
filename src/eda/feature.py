import transformers

def addTokenLength(data, config):
    tokenizer = transformers.AutoTokenizer.from_pretrained(config["model"]["name"], max_length=320)
    
    data['tokenLength_1'] = data['sentence_1'].apply(lambda x: len(tokenizer(x)['input_ids']))
    data['tokenLength_2'] = data['sentence_2'].apply(lambda x: len(tokenizer(x)['input_ids']))
    
    return data