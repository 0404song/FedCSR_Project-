# src/server.py
import torch
import numpy as np
import logging
from tqdm import tqdm

# Import all aggregators from our new module
from src.aggregators import FedAvgAggregator, FedNormAggregator, FedCSRHardFilter, FedCSRDirectional
# Add others here later, e.g., FedCSRDirectional

class Server:
    def __init__(self, model, dataset, config):
        self.model = model
        self.dataset = dataset
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.device = torch.device(self.config.get('device', 'cpu'))
        self.model.to(self.device)

        # A "factory" dictionary to map config strings to aggregator classes
        aggregator_factory = {
            'fedavg': FedAvgAggregator,
            'fednorm': FedNormAggregator,
            'fedcsr_hardfilter': FedCSRHardFilter,
            'fedcsr_directional': FedCSRDirectional, # We will add this later
        }

        agg_method = self.config.get('defense_method', 'fedavg')
        aggregator_class = aggregator_factory.get(agg_method)

        if aggregator_class:
            self.aggregator = aggregator_class(self.config, self.logger)
            self.logger.info(f"Server initialized with '{agg_method}' aggregation strategy.")
        else:
            raise ValueError(f"Unknown defense_method: {agg_method}. Available: {list(aggregator_factory.keys())}")

    def select_clients(self, all_client_ids):
        fraction = self.config.get('client_fraction', 0.1)
        num_to_select = int(len(all_client_ids) * fraction)
        return np.random.choice(all_client_ids, size=num_to_select, replace=False)

    def aggregate_updates(self, client_updates, selected_clients_indices):
        # The server now simply delegates the aggregation task
        return self.aggregator.aggregate(client_updates, selected_clients_indices)

    def update_model(self, aggregated_update):
        if aggregated_update is None:
            self.logger.info("Skipping model update as aggregated_update is None.")
            return
        current_state = self.model.state_dict()
        for key in current_state:
            # Note: += might cause in-place modification issues, this is safer
            current_state[key] = current_state[key] + aggregated_update[key].to(self.device)
        self.model.load_state_dict(current_state)

    def evaluate(self):
        self.model.eval()
        all_user_ids = list(self.dataset.test_ground_truth.keys())
        recalls, ndcgs = [], []
        k = self.config.get('eval_k', 10)

        with torch.no_grad():
            for user_id in tqdm(all_user_ids, desc="Evaluating", leave=False):
                user_tensor = torch.LongTensor([user_id]).to(self.device)
                all_item_ids = torch.arange(self.dataset.num_items).to(self.device)
                
                # Check for potential empty user tensor
                if user_tensor.shape[0] == 0: continue

                user_tensor_expanded = user_tensor.expand(self.dataset.num_items)
                
                scores = self.model.predict(user_tensor_expanded, all_item_ids)
                
                train_items = self.dataset.train_data.get(user_id, [])
                if train_items:
                    scores[train_items] = -np.inf
                
                _, top_k_items = torch.topk(scores, k=k)
                top_k_items = top_k_items.cpu().numpy()
                
                true_items = self.dataset.test_ground_truth.get(user_id, [])
                if not true_items:
                    recalls.append(0)
                    ndcgs.append(0)
                    continue
                
                hits = [1 if item in true_items else 0 for item in top_k_items]
                
                recall_score = sum(hits) / len(true_items)
                recalls.append(recall_score)

                dcg = sum([h / np.log2(i + 2) for i, h in enumerate(hits)])
                idcg = sum([1 / np.log2(i + 2) for i in range(min(len(true_items), k))])
                ndcg_score = dcg / idcg if idcg > 0 else 0
                ndcgs.append(ndcg_score)
                
        return np.mean(recalls), np.mean(ndcgs)
