# src/model.py (Final Version)
import torch
import torch.nn as nn

class MFModel(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim):
        super(MFModel, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # Initialize embeddings with a normal distribution
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)

    def forward(self, user_ids, item_ids):

        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        
        # Dot product for positive interaction scores
        pos_scores = torch.mul(user_emb, item_emb).sum(dim=1)
        
        return pos_scores

    def predict(self, user_ids, item_ids):

        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        scores = torch.mul(user_emb, item_emb).sum(dim=1)
        return scores

    def get_update(self, old_model_state):
        #"""Calculates the difference between current model state and an old one."""
        update = {}
        for key, new_param in self.state_dict().items():
            old_param = old_model_state[key]
            update[key] = new_param - old_param
        return update
