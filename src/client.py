# src/client.py (最终架构重构版)

import torch
import torch.optim as optim
import random
from copy import deepcopy
from .model import MatrixFactorization, bpr_loss

class Client:
    def __init__(self, client_id, local_train_data, all_item_ids, num_items, device):
        # The Client is now lightweight, only holding data and ID.
        self.id = client_id
        self.local_train_data = set(local_train_data)
        self.all_item_ids = all_item_ids
        self.num_items = num_items
        self.device = device
        # NO self.model or self.optimizer here.

    def negative_sampling(self, pos_items):
        neg_items = []
        for _ in pos_items:
            neg_item = random.choice(list(self.all_item_ids))
            while neg_item in self.local_train_data:
                neg_item = random.choice(list(self.all_item_ids))
            neg_items.append(neg_item)
        return neg_items

    def train(self, model_template, local_epochs=5, batch_size=32, lr=0.01):
        # --- The entire lifecycle of the local model and optimizer is contained in this function ---
        
        # 1. Create temporary local model and optimizer
        local_model = deepcopy(model_template).to(self.device)
        optimizer = optim.Adam(local_model.parameters(), lr=lr)
        
        local_model.train()
        global_params_cpu = {k: v.cpu() for k, v in model_template.state_dict().items()}

        for epoch in range(local_epochs):
            pos_items = list(self.local_train_data)
            random.shuffle(pos_items)
            
            for i in range(0, len(pos_items), batch_size):
                batch_pos_items = pos_items[i:i + batch_size]
                batch_neg_items = self.negative_sampling(batch_pos_items)
                
                user_indices = torch.LongTensor([self.id] * len(batch_pos_items)).to(self.device)
                pos_item_indices = torch.LongTensor(batch_pos_items).to(self.device)
                neg_item_indices = torch.LongTensor(batch_neg_items).to(self.device)

                optimizer.zero_grad()
                pos_scores, neg_scores = local_model(user_indices, pos_item_indices, neg_item_indices)
                loss = bpr_loss(pos_scores, neg_scores)
                
                loss.backward()
                optimizer.step()
        
        # 2. Calculate update and move to CPU
        local_params_cpu = {k: v.cpu() for k, v in local_model.state_dict().items()}
        model_update = {key: local_params_cpu[key] - global_params_cpu[key] for key in global_params_cpu.keys()}
            
        # 3. local_model and optimizer go out of scope and are destroyed.
        return model_update

