import torch
import torchvision
import torch.nn.functional as F
from torch import nn
from utils import pairselector



def CrossEntropy():
    """Cross Entropy Loss"""
    loss = torch.nn.CrossEntropyLoss()
    return loss

class ContrastiveLoss(nn.Module):
    """Contrastive loss. Takes vectors and labels as 1 and 0. output1 and output2 are normalised with L2 norm.
     Based on http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf"""
    def __init__(self, margin = 2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9

    def forward(self, output1, output2, label):
        distance = torch.sqrt(torch.pow((F.normalize(output2, p=2, dim=1) - F.normalize(output1, p=2, dim=1)), 2).sum(1))
        loss = 0.5*(label.float())*torch.pow(distance, 2) + (1-label.float())*0.5*torch.pow(F.relu(self.margin - (distance + self.eps)), 2)
        return loss.mean()

class OnlineContrastiveloss(nn.Module):
    """A variant of Contrastive loss, that generates loss from the embeddings. Embeddings are normalised with L2 norm. This doesn't need Siamese network."""
    def __init__(self, margin, pairselector):
        super(OnlineContrastiveloss, self).__init__()
        self.eps = 1e-9
        self.margin = margin
        self.pairselector = pairselector

    def forward(self, embeddings, labels):
        pos_pairs, neg_pairs = self.pairselector.get_pairs(embeddings, labels)
        pos_pairs = pos_pairs.cuda()
        neg_pairs = neg_pairs.cuda()
        pos_loss = torch.pow((F.normalize(embeddings[pos_pairs[:,0]], p=2, dim=1) - F.normalize(embeddings[pos_pairs[:,1]], p=2, dim=1)), 2).sum(1)
        neg_loss = torch.pow(F.relu(self.margin - (torch.sqrt(torch.pow((F.normalize(embeddings[neg_pairs[:,0]], p=2, dim=1) - F.normalize(embeddings[neg_pairs[:,1]], p=2, dim=1)), 2).sum(1)) + self.eps)), 2)
        loss = torch.cat([pos_loss, neg_loss], dim = 0)
        return loss.mean()