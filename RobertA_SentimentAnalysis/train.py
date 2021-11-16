import csv
import torch
import numpy as np
import pandas as pd
import os
import tqdm
from tqdm import tqdm
from transformers import RobertaTokenizer
from sklearn.metrics import f1_score
import random
from transformers import  RobertaModel
from torch.utils.data import TensorDataset, DataLoader





def f1_score_func(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, preds_flat, average='weighted')


def evaluate(model,dataloader, device):

    model.eval()
    total_loss = 0
    predictions, true_labels = [], []
    criterion = torch.nn.CrossEntropyLoss()

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
        loss = criterion(outputs, targets)
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

def main():
    # If there's a GPU available...
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")

    # set the seeds
    seed_val = 10
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)


    #read the files
    df = pd.read_csv('dataset_raw.csv')
    df_valid = pd.read_csv('validation_raw.csv')
    df_test = pd.read_csv('Test.csv')

    # instanciate the tokeniser
    roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base',truncation=True)

    #go through each text value in the train dataset and tokenise them and save the some attributes
    input_ids_train = [];attention_masks_train = []
    token_type_ids_train = [];labels_train = []
    labels_train = []

    print("I am going to encode train data")
    for i in range(len(df.text)):

        text = str(df['text'][i])
        encoded_train_data = roberta_tokenizer.encode_plus(
            text,
            text_pair=None,
            add_special_tokens=True, # Add '[CLS]' and '[SEP]'
            max_length=256, #truncate all sentences.
            pad_to_max_length=True, # Pad all sentences
            return_token_type_ids=True #return token IDs
        )

        input_ids_train.append(encoded_train_data['input_ids'])
        attention_masks_train.append(encoded_train_data['attention_mask'])
        token_type_ids_train.append(encoded_train_data['token_type_ids'])
        labels_train.extend(torch.tensor(df['label'][i]).unsqueeze(dim=0))

        # go through each text value in the validation dataset and tokenise them and save the some attributes
        input_ids_val = []
        attention_masks_valid = []
        token_type_ids_valid = []
        labels_valid = []
    print("I am going to encode valid data")
    for i in range(len(df_valid.text)):

        text_valid = df_valid['text'][i]
        encoded_valid_data = roberta_tokenizer.encode_plus(
            text=text_valid,
            text_pair=None,
            add_special_tokens=True,
            max_length=256,
            pad_to_max_length=True,
            return_token_type_ids=True
        )

        input_ids_val.append(encoded_valid_data['input_ids'])
        attention_masks_valid.append(encoded_valid_data['attention_mask'])
        token_type_ids_valid.append(encoded_valid_data['token_type_ids'])
        labels_valid.append(torch.tensor(df_valid['label'][i]).unsqueeze(dim=0))

    #  build  final torch dataset
    dataset_train = TensorDataset(torch.tensor(input_ids_train), torch.tensor(attention_masks_train),
                                  torch.tensor(token_type_ids_train), torch.tensor(labels_train))
    dataset_val = TensorDataset(torch.tensor(input_ids_val), torch.tensor(attention_masks_valid),
                                torch.tensor(token_type_ids_valid), torch.tensor(labels_valid))


    # bulid the dataloaders
    batch_size = 16
    dataloader_train = DataLoader(
        dataset_train,
        shuffle=True,
        batch_size=batch_size
    )

    dataloader_val = DataLoader(
        dataset_val,
        shuffle=True,
        batch_size=16
    )


    # add classificaton layer to RoberaModel
    class Roberta(torch.nn.Module):
        def __init__(self):
            super(Roberta, self).__init__()
            self.hidden_states = RobertaModel.from_pretrained("roberta-base")
            self.layer1 = torch.nn.Linear(768, 768)
            self.relu = torch.nn.ReLU()
            self.dropout = torch.nn.Dropout(0.5)
            self.layer2 = torch.nn.Linear(768, 2)
            self.prob = torch.nn.Softmax(dim=1)

        def forward(self, input_ids, attention_mask, token_type_ids):

            hidden_state = self.hidden_states(input_ids=input_ids, attention_mask=attention_mask,
                                          token_type_ids=token_type_ids)[0]
            z = hidden_state[:, 0]
            z = self.layer1(z)
            z = self.relu(z)
            z = self.dropout(z)
            output = self.layer2(z)
            out = self.prob(output)
            return out


    model = Roberta()
    model.to(device)
    # Creating the loss function and optimizer
    LEARNING_RATE = 1e-5
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

    epoch = 5
    print("I am about to enter to the training loop")
    for j in range(1, epoch + 1):

        model.train()
        for iteration, data in tqdm(enumerate(dataloader_train, 0)):

            ids = data[0].to(device, dtype=torch.long)
            mask = data[1].to(device, dtype=torch.long)
            token_type_ids = data[2].to(device, dtype=torch.long)
            targets = data[3].to(device, dtype=torch.long)

            outputs = model(ids, mask, token_type_ids)
            loss = criterion(outputs, targets)

            if iteration % 2000 == 0:
                valid_loss, predictions_valid, true_valid = evaluate(model, dataloader_val, device)
                valid_f1 = f1_score_func(predictions_valid, true_valid)
                train_loss, predictions_train, true_train = evaluate(model, dataloader_train, device)
                train_f1 = f1_score_func(predictions_train, true_train)
                print("\nf1 score on valid data is: ", valid_f1)
                print("\nf1 score on train data is: ", train_f1)


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        #print("iteration is: ", iteration)
        #torch.save(model.state_dict(), f'RoBERTa_epoch{j}.model')
        torch.save(model.state_dict(), f'RoBERTa_M_epoch{j}.model')
        tqdm.write(f'\nEpoch {j}')


if __name__ == "__main__":
    main()




