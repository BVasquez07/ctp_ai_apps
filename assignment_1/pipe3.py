from transformers import pipeline

classifier = pipeline("zero-shot-classification")

res = classifier(
    "This is a review on a brand new cracker",
    candidate_labels=['games','snack','food','kitchen supply']
)
print(res)