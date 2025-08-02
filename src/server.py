# src/server.py (最终架构重构版)

import torch
from tqdm import tqdm
import random
from .model import MatrixFactorization
from .client import Client
from .metrics import recall_at_k, ndcg_at_k
import gc

class Server:
    def __init__(self, num_users, num_items, train_data, test_data, embedding_dim, device): # <--- 添加
        self.device = device
        self.global_model = MatrixFactorization(num_users, num_items, embedding_dim).to(device) # <
        self.num_users = num_users
        self.num_items = num_items
        
        all_item_ids = set(range(num_items))
        self.clients = []
        for client_id in range(num_users):
            if client_id in train_data:
                client = Client(client_id, train_data[client_id], all_item_ids, num_items, device)
                self.clients.append(client)
        
        self.test_data = test_data

    def select_clients(self, fraction=0.1):
        num_selected = max(1, int(len(self.clients) * fraction))
        return random.sample(self.clients, num_selected)

    def aggregate_updates(self, updates):
        if not updates: return
        keys = self.global_model.state_dict().keys()
        with torch.no_grad():
            aggregated_update = {key: torch.zeros_like(updates[0][key]) for key in keys}
            for key in keys:
                update_stack = torch.stack([upd[key] for upd in updates], dim=0)
                aggregated_update[key] = torch.mean(update_stack, dim=0)
            current_params = self.global_model.state_dict()
            for key in keys:
                current_params[key] += aggregated_update[key].to(self.device)
            self.global_model.load_state_dict(current_params)
        
    def train_round(self, num_round, client_fraction=0.1, local_epochs=5, lr=0.01):
        print(f"\n--- Training Round {num_round} ---")
        selected_clients = self.select_clients(fraction=client_fraction)
        print(f"Selected {len(selected_clients)} clients for this round.")
        
        updates = []
        # The loop is now clean. It doesn't modify the client objects.
        for client in tqdm(selected_clients, desc="Clients training"):
            # The client is just a "worker" that takes the model template and returns an update.
            update = client.train(self.global_model, local_epochs=local_epochs, lr=lr)
            updates.append(update)
            
        print("Aggregating updates...")
        self.aggregate_updates(updates)

        # Explicitly clean up to be safe
        del updates
        gc.collect()
        torch.cuda.empty_cache()

    def evaluate(self, k=10):
        self.global_model.eval()
        total_recall, total_ndcg, num_test_clients = 0, 0, 0
        all_item_indices = torch.arange(self.num_items).to(self.device)
        with torch.no_grad():
            for client in tqdm(self.clients, desc="Evaluating"):
                true_items = self.test_data.get(client.id, [])
                if not true_items: continue
                num_test_clients += 1
                user_index = torch.LongTensor([client.id] * self.num_items).to(self.device)
                user_embedding = self.global_model.user_embeddings(user_index)
                item_embeddings = self.global_model.item_embeddings(all_item_indices)
                scores = torch.sum(user_embedding * item_embeddings, dim=1)
                scores[list(client.local_train_data)] = -float('inf')
                total_recall += recall_at_k(true_items, scores, k)
                total_ndcg += ndcg_at_k(true_items, scores, k)
        avg_recall = total_recall / num_test_clients if num_test_clients > 0 else 0
        avg_ndcg = total_ndcg / num_test_clients if num_test_clients > 0 else 0
        print(f"\n--- Evaluation Results ---")
        print(f"Recall@{k}: {avg_recall:.4f}")
        print(f"NDCG@{k}: {avg_ndcg:.4f}")
        print(f"--------------------------\n")
        return avg_recall, avg_ndcg
