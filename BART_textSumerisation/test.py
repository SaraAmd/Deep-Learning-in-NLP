import os
import csv
from transformers import BartTokenizer, BartForConditionalGeneration

from torch.utils.data import TensorDataset, DataLoader
import torch
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

from transformers import Trainer, TrainingArguments

import pandas as pd
import random
from rouge import  Rouge
import numpy as np
import datasets

from torch.utils.data import TensorDataset, DataLoader
from transformers import BartTokenizer, BartForConditionalGeneration


def main():
    # for reproducibility
    seed_val = 10
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cuda")
     #load dataset
    test_data = datasets.load_dataset("xsum", split="test")
    #load tokeniser
    bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

    def data_preprocess(example_batch):

        input_encodings=bart_tokenizer.batch_encode_plus(
            example_batch['document'],
            max_length=1024,
            pad_to_max_length=True,
            truncation=True,
            return_tensors='pt'
        )
        target_encodings = bart_tokenizer.batch_encode_plus(
            example_batch['summary'], pad_to_max_length=True,
            max_length=1024, truncation=True, return_tensors='pt')

        labels = target_encodings['input_ids']
        input_ids= input_encodings['input_ids']
        attention_mask= input_encodings['attention_mask']
        encodings = {
            'input_ids': list(input_encodings['input_ids']),
            'attention_mask': list(input_encodings['attention_mask']),
            'labels': list(labels)
        }
        return encodings


    dataset = test_data.map(data_preprocess, batched=True)
    columns = ['input_ids', 'labels', 'attention_mask' ]
    dataset.set_format(type='torch', columns=columns)

    # Load trained model
    model_path = "models/bart-summarizer/checkpoint-204000"
    model = BartForConditionalGeneration.from_pretrained(model_path, local_files_only=True)
    model.to(device)

    rough = Rouge()
    rough_score = []
    for i in range(len(test_data['document'])):
        input_tokenized = bart_tokenizer.encode(test_data['document'][i], pad_to_max_length=True,
            max_length=1024, truncation=True, return_tensors='pt').to(device)

        summary_ids = model.generate(input_tokenized.to(device),
                                      num_beams=4,
                                      num_return_sequences=1,
                                      no_repeat_ngram_size=2,
                                      length_penalty=1,
                                      min_length=12,
                                      max_length=56,
                                     #max_length=142,
                                      early_stopping=True)
        output = [bart_tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in
              summary_ids]
        #print("the generated output is: ", output)

        score = rough.get_scores(output[0], test_data['summary'][i])
        rough_score.append(score[0]['rouge-2']['f'])
        print("f1_score is; ", score[0]['rouge-2']['f'])
    average = sum(rough_score)/len(rough_score)
    print("average is: ", average)



if __name__ == "__main__":
    main()