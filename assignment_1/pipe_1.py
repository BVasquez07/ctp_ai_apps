from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
classifier = pipeline("sentiment-analysis") #task of the classifier

res = classifier("Hello I am kind of funny are you?") #data to pass into the classifier for doing sentiment-analysis on

data = [
    "I feel sad",
    "I feel happy"
]
for i,j in enumerate(data):
    print(i, j)

print(res, "first classification without tokenizer and model explicitly stated")


model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer) #task of the classifier

res = classifier(data) #data to pass into the classifier for doing sentiment-analysis on



print(res)