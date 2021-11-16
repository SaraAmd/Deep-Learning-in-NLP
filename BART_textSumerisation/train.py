import os
import csv
from transformers import BartTokenizer, BartForConditionalGeneration
#from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from torch.utils.data import TensorDataset, DataLoader
import torch
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
#from transformers import TrainingArguments, HfArgumentParser
from transformers import Trainer, TrainingArguments

import time

import pandas as pd
import  pdb

import datasets
from dataset import LabeledDataset
from torch.utils.data import TensorDataset, DataLoader

from transformers import BartTokenizer, BartForConditionalGeneration
from transformers.models.bart.modeling_bart import shift_tokens_right

from rouge import  Rouge

def main():

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cuda")

    #load the dataset
    train_data = datasets.load_dataset("xsum", split="train")
    val_data = datasets.load_dataset("xsum", split="validation")

   #get the model
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-base").to(device)
    bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

    #define a preprocessing function
    
    def data_preprocess(example_batch):

        input_encodings=bart_tokenizer.batch_encode_plus(
            example_batch['document'],
            max_length=1024,
            pad_to_max_length=True,
            truncation=True,
            return_tensors='pt',
            add_special_tokens= True
            
        )
        target_encodings = bart_tokenizer.batch_encode_plus(
            example_batch['summary'],
            pad_to_max_length=True,
            max_length=1024,
            truncation=True,
            return_tensors='pt',
	    add_special_tokens= True

            

        )

        labels = target_encodings['input_ids']
        input_ids= input_encodings['input_ids']
        attention_mask= input_encodings['attention_mask']

        encodings = {
            'input_ids': list(input_encodings['input_ids']),
            'attention_mask': list(input_encodings['attention_mask']),
            'labels': list(labels)
        }
        return encodings


    #apply the data preprocessing on the datatset
    dataset = train_data.map(data_preprocess, batched=True)
    dataset_val= val_data.map(data_preprocess, batched=True)

   #make the final dataset for training
    columns = ['input_ids', 'labels', 'attention_mask' ]
    dataset.set_format(type='torch', columns=columns)
    dataset_val.set_format(type='torch', columns=columns)

    training_args = TrainingArguments(
        # The output directory where the model
        #  predictions and checkpoints will be written.
        output_dir='./models/bart-summarizer',
        # Overwrite the content of the output directory.
        overwrite_output_dir=True,
        # Whether to run training or not.
        do_train=True,
        # Whether to run evaluation on the dev or not.
        do_eval=True,
        # Total number of training epochs to perform
        num_train_epochs=1,
        # Batch size GPU/TPU core/CPU training.
        per_device_train_batch_size=1,
        # Batch size  GPU/TPU core/CPU for evaluation.
            per_device_eval_batch_size=8,
            warmup_steps=1000, # number of warmup steps for learning rate
            weight_decay=0.01,
            logging_dir='./logs',
        # evaluation strategy to adopt during training
        # `no`: No evaluation during training.
        # `steps`: Evaluate every `eval_steps`.
        # `epoch`: Evaluate every end of epoch.
        evaluation_strategy="steps",
        # Number of update steps between two
        # evaluations if evaluation_strategy="steps".
        # Will default to the same value as l
        # logging_steps if not set.

        eval_steps=4000,  # Evaluation and Save happens every 4000 steps
        save_steps=4000,
        # How often to show logs. I will se this to
        # plot history loss and calculate perplexity.
        logging_steps=700,
        save_total_limit=5,  # Only last 5 models are saved. Older ones are deleted.
        load_best_model_at_end=True
    )

    # Define Trainer parameters
    def compute_metrics(p):
        pred, labels = p
        pred = np.argmax(pred, axis=1)

        #accuracy = accuracy_score(y_true=labels, y_pred=pred)
        #recall = recall_score(y_true=labels, y_pred=pred)
        #precision = precision_score(y_true=labels, y_pred=pred)
        f1 = f1_score(y_true=labels, y_pred=pred)

        return { "f1": f1}

    trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            eval_dataset=dataset_val,
        #compute_metrics=compute_metrics
    )
    trainer.train()
    trainer.save_model("./bart-summerizer")
    
    model.save_pretrained("Trained model")  # did save


if __name__ == "__main__":
    main()