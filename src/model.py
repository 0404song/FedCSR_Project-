# src/model.py
import torch
import torch.nn as nn

class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=32): # <--- Ìí¼Ó embedding_dim
        super(MatrixFactorization, self).__init__()
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)
        nn.init.normal_(self.user_embeddings.weight, std=0.01)
        nn.init.normal_(self.item_embeddings.weight, std=0.01)

    def forward(self, user_indices, pos_item_indices, neg_item_indices):
        user_embedding = self.user_embeddings(user_indices)
        pos_item_embedding = self.item_embeddings(pos_item_indices)
        neg_item_embedding = self.item_embeddings(neg_item_indices)
        pos_scores = torch.sum(user_embedding * pos_item_embedding, dim=1)
        neg_scores = torch.sum(user_embedding * neg_item_embedding, dim=1)
        return pos_scores, neg_scores

def bpr_loss(pos_scores, neg_scores):
    return -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8))
