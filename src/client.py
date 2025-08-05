# src/client.py (Final Version)
import torch
import copy
import random
from torch.utils.data import DataLoader
from src.dataset import LocalDataset

class Client:
    def __init__(self, client_id, model, dataset, config, is_malicious=False):
        self.client_id = client_id
        self.model = model
        self.dataset = dataset
        self.config = config
        self.is_malicious = is_malicious
        self.device = torch.device(self.config.get('device', 'cpu'))
        self.model.to(self.device)
        self.learning_rate = self.config.get('learning_rate', 0.01)
        self.local_epochs = self.config.get('local_epochs', 5)
        self.batch_size = self.config.get('batch_size', 32)
        
        self.attack_logic = self.config.get('attack_logic', 'sign_flipping')
        self.amp_factor = self.config.get('attack_amplification_factor', 5.0)

    def train(self):
        #"""Trains the client's model on its local data."""
        client_data = self.dataset.get_client_data(self.client_id)
        if not client_data:
            return None # Skip training if no data

        local_dataset = LocalDataset(self.client_id, client_data)
        data_loader = DataLoader(local_dataset, batch_size=self.batch_size, shuffle=True)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        initial_state = copy.deepcopy(self.model.state_dict())

        self.model.train()
        for _ in range(self.local_epochs):
            for user_ids, pos_item_ids in data_loader:
                user_ids, pos_item_ids = user_ids.to(self.device), pos_item_ids.to(self.device)
                
                # BPR Loss Calculation
                neg_item_ids = torch.randint(0, self.dataset.num_items, (len(user_ids),)).to(self.device)
                
                pos_scores = self.model(user_ids, pos_item_ids)
                neg_scores = self.model(user_ids, neg_item_ids)
                
                loss = -torch.nn.functional.logsigmoid(pos_scores - neg_scores).mean()
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        update = self.model.get_update(initial_state)

        if self.is_malicious:
            update = self._apply_attack(update)
            
        return update

    def _apply_attack(self, update):
        #"""Applies a model poisoning attack to the update."""
        if self.attack_logic == 'sign_flipping':
            for key in update:
                update[key] = -self.amp_factor * update[key]
        # Add other attack logics here if needed
        return update
