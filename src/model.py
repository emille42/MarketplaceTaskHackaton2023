import torch.nn as nn
import torch

class BertSiameseModel(nn.Module):
  def __init__(self, bert):
    super().__init__()
    self.bert = bert

    self.maxpool = nn.AdaptiveMaxPool1d(output_size=1)
    self.maxpool2 = nn.AdaptiveMaxPool1d(output_size=64)
    self.conv = nn.Conv1d(4,1, kernel_size=3, padding=1)

    self.bn = nn.BatchNorm1d(64)
    self.relu = nn.ReLU()
    self.linear = nn.Linear(64, 32)
    self.bn1 = nn.BatchNorm1d(32)
    self.linear2 = nn.Linear(32, 16)
    self.bn2 = nn.BatchNorm1d(16)
    self.linear3 = nn.Linear(16, 2)
    self.sm_out = nn.Softmax(-1)
    


  def forward(self, ids_1, type_ids_1,ids_2, type_ids_2, name_1, name_2, att_1=None, att_2=None):
    
    # states: batch x seq_len x emb_size
    h_state_1 = self.bert(input_ids=ids_1, token_type_ids=type_ids_1, attention_mask=att_1)['last_hidden_state']
    h_state_2 = self.bert(input_ids=ids_2, token_type_ids=type_ids_2, attention_mask=att_2)['last_hidden_state']
    
    # batch x 64
    h_state_1_vector = self.maxpool2(self.maxpool(h_state_1.permute(0,2,1)).squeeze())
    h_state_2_vector = self.maxpool2(self.maxpool(h_state_2.permute(0,2,1)).squeeze())
    
    # batch x 4 x 64
    stacked = torch.stack([h_state_1_vector, h_state_2_vector, name_1, name_2], 1)

    # batch x 64
    stacked = self.conv(stacked).squeeze()

    stacked = self.bn(stacked)
    stacked = self.relu(stacked)

    x = self.linear(stacked)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.linear2(x)
    x = self.bn2(x)
    x = self.relu(x)
    out = self.linear3(x)
    
    out = self.sm_out(out)
    out = out.T[1]
    out = out.squeeze()
    return out    