
import numpy as np
import evaluate
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer


def prepare_dataset():
    raw_datasets = load_dataset("imdb")
    return raw_datasets

def tokenize_function(example):   
    checkpoint = "bert-base-cased" 
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    return tokenizer(
        example["text"], 
        padding="max_length",
        truncation=True,
        max_length=128
    )
    
def compute_metrics(eval_pred):
    metric = evaluate.load("accuracy")    
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)
    
def fine_tune(raw_datasets):    
    tokenized_dataset = raw_datasets.map(tokenize_function, batched=True)   
    full_train_ds = tokenized_dataset["train"]
    full_eval_ds = tokenized_dataset["test"]
    
    checkpoint = "bert-base-cased"
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
    
    training_args = TrainingArguments(
        output_dir="ft_model",
        eval_strategy="epoch",
        num_train_epochs=5        
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=full_train_ds,
        eval_dataset=full_eval_ds,
        compute_metrics=compute_metrics
    )
    trainer.train()