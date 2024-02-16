from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from pathlib import Path

data = [
    "I feel sad",
    "I feel happy"
]
for i,j in enumerate(data):
    print(i, j)



model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer) #task of the classifier

res = classifier(data) #data to pass into the classifier for doing sentiment-analysis on

print(res)


batch = tokenizer(data, padding=True, truncation=True, max_length=512, return_tensors="pt")

with torch.no_grad():
    outputs = model(**batch)
    print(outputs)
    predictions = F.softmax(outputs.logits, dim=1)
    print(predictions)
    labels= torch.argmax(predictions, dim=1)
    print(labels)

save_dir = Path.cwd()
tokenizer.save_pretrained(save_dir) #save tokenizer
model.save_pretrained(save_dir) #save model

#tok = AutoTokenizer.save_pretrained(save_dir) #load tokenizer
#mod = AutoModelForSequenceClassification.save_pretrained(save_dir) #load model
