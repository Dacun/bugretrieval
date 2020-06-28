import torch.nn as nn
import torch.nn.functional as F

from data_generator import *
from utils import load_emb_matrix


class CNN_Text(nn.Module):
  def __init__(self, input_dim, n_filters):
    super(CNN_Text, self).__init__()
    D = input_dim
    Ci = 1
    Co = n_filters
    Ks = [3, 4, 5]
    self.convs = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
    self.fc = nn.Sequential(nn.Linear(n_filters * 3, 100), nn.Tanh())

  def forward(self, x):
    x = x.unsqueeze(1)
    x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
    x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
    x = torch.cat(x, 1)
    return self.fc(x)


class BaseNet(torch.nn.Module):
  def __init__(self, args):
    super(BaseNet, self).__init__()
    self.word_embed = nn.Embedding(args.n_words, args.word_dim, max_norm=1, padding_idx=0)
    self.word_embed.weight = nn.Parameter(
      torch.from_numpy(load_emb_matrix(args.n_words, args.word_dim, args.data)).float()
    )
    
    self.CNN = CNN_Text(args.word_dim, args.n_filters)
    self.RNN = nn.GRU(input_size=args.word_dim, hidden_size=50, bidirectional=True, batch_first=True)

    self.info_proj = nn.Sequential(nn.Linear(1123, 100), nn.Tanh())
    self.projection = nn.Linear(300, 100)


  def forward(self, x):
    # x = [info, desc, short desc]
    info = x['info']
    info_feature = self.info_proj(info.float())

    desc = x['desc'][0]
    desc_feature = self.CNN(self.word_embed(desc))

    short_desc = x['short_desc'][0]
    out, hidden = self.RNN(self.word_embed(short_desc))
    short_desc_feature = torch.mean(out, dim=1)
    
    feature = torch.cat([info_feature, short_desc_feature, desc_feature], -1)
    return self.projection(feature)


class MarginLoss(torch.nn.Module):
  def __init__(self, margin=1.0):
    super(MarginLoss, self).__init__()
    self.margin = margin
    self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

  def forward(self, x, x_pos, x_neg):
    fb1 = self.cos(x, x_pos)
    fb2 = self.cos(x, x_neg)
    loss = self.margin - fb1 + fb2
    loss = F.relu(loss)
    return loss.mean()