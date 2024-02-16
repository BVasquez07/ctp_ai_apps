from transformers import AutoTokenizer 

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

sequence = "These are some words!"
res = tokenizer(sequence)
print(res)
tokens = tokenizer.tokenize(sequence) #convert ourt str to mathematical representation for the model to understand
print(tokens)
ids = tokenizer.convert_tokens_to_ids(tokens)
print(ids)
decoded_str = tokenizer.decode(ids)
print(decoded_str)