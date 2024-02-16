"""
Training workflow..

1. prepare dataset
2. load pretrained tokenizer, call it with dataset (encoding)
3. build PyTorch dataset with encodings 
4. load pretrained model
5. a) load Trainer and train it
   b) native PyTorch training loop

"""

from transformers import Trainer, TrainingArguments

training_args = TrainingArguments("test-trainer")

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer
)

trainer.train()