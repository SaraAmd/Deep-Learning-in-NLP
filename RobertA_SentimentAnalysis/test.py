import csv
import torch
import numpy as np
import pandas as pd
import os
import tqdm
from tqdm import tqdm
from transformers import RobertaTokenizer
from transformers import BertTokenizer
from sklearn.metrics import f1_score
import random
from transformers import RobertaModel
from model import Roberta
from torch.utils.data import TensorDataset, DataLoader

def f1_score_func(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, preds_flat, average='weighted')



def evaluate(model,dataloader, device):

    model.eval()
    total_loss = 0
    predictions, true_labels = [], []
    Loss = torch.nn.CrossEntropyLoss()

    for batch in dataloader:
        batch = tuple(b.to(device) for b in batch)

        inputs = {
            'input_ids': batch[0],
            'attention_mask': batch[1],
            'token_type_ids':batch[2],
        }
        with torch.no_grad():
            outputs = model(**inputs)

        targets = batch[3].to(device, dtype=torch.long)
        loss = Loss(outputs, targets)
        total_loss += loss.item()
        logits = outputs.data
        logits = logits.detach().cpu().numpy()
        label_ids = batch[3].cpu().numpy()
        predictions.append(logits)
        true_labels.append(label_ids)

    loss_avg = total_loss / len(dataloader)
    predictions = np.concatenate(predictions, axis=0)
    true_labels = np.concatenate(true_labels, axis=0)

    return loss_avg, predictions, true_labels

def reproduce():

    # for reproducibility
    seed_val = 10
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    # If there's a GPU available...
    if torch.cuda.is_available():
        # Tell PyTorch to use the GPU.
        device = torch.device("cuda")
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")

    df_test = pd.read_csv('Test.csv')
    # instanciate the tokeniser
    roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base', truncation=True)
    input_ids_test = []
    attention_masks_test = []
    token_type_ids_test = []
    labels_test = []
    print("I am going to encode test data")
    for i in range(len(df_test.text)):

        text_test = (df_test['text'][i])

        encoded_test_data = roberta_tokenizer.encode_plus(
            text=text_test,
            text_pair=None,
            add_special_tokens=True,
            max_length=256,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        input_ids_test.append(encoded_test_data['input_ids'])
        attention_masks_test.append(encoded_test_data['attention_mask'])
        token_type_ids_test.append(encoded_test_data['token_type_ids'])
        labels_test.append(torch.tensor(df_test['label'][i]).unsqueeze(dim=0))

    #  build  final torch dataset

    dataset_test = TensorDataset(torch.tensor(input_ids_test), torch.tensor(attention_masks_test),
                                 torch.tensor(token_type_ids_test), torch.tensor(labels_test))

    dataloader_test = DataLoader(
        dataset_test,
        shuffle=True,
        batch_size=16
    )
    model = Roberta()
    model.load_state_dict(torch.load('RoBERTa_M_epoch5.model'))
    model.to(device)

    test_loss, predictions_test, true_test = evaluate(model, dataloader_test, device)
    test_f1 = f1_score_func(predictions_test, true_test)
    print(f'Test f1 score: {test_f1:.3f}')




if __name__ == "__main__":
    reproduce()




