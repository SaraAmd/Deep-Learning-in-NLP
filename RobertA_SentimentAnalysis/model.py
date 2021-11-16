from transformers import RobertaTokenizer
from transformers import   RobertaModel
import torch

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