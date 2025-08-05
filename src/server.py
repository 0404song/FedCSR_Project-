# src/server.py
import torch
import numpy as np
import logging
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

# Import all aggregators from our new module
from src.aggregators import FedAvgAggregator, FedNormAggregator, FedCSRHardFilter, FedCSRDirectional,FedRepAggregator
# Add others here later, e.g., FedCSRDirectional

# In src/server.py, update the Server.__init__ method


class Server:
    def __init__(self, model, dataset, config):
        self.model = model
        self.dataset = dataset
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.device = torch.device(self.config.get('device', 'cpu'))
        self.model.to(self.device)

        # --- Client Reputation Management (Core of FedRep) ---
        # Initialize reputation for all clients. We have num_users from the dataset.
        self.client_reputations = {i: 1.0 for i in range(self.dataset.num_users)}
        self.proxy_dataloader = None # Will be initialized if needed by aggregator

        aggregator_factory = {
            'fedavg': FedAvgAggregator,
            'fednorm': FedNormAggregator,
            'fedcsr_directional': FedCSRDirectional,
            # Add our new FedRep aggregator
            'fedrep': FedRepAggregator,
        }
        
        agg_method = self.config.get('defense_method', 'fedavg')
        aggregator_class = aggregator_factory.get(agg_method)
        if aggregator_class:
            # Pass server instance to aggregator so it can access reputations
            self.aggregator = aggregator_class(self.config, self.logger, server_instance=self) 
            self.logger.info(f"Server initialized with '{agg_method}' aggregation strategy.")
        else:
            raise ValueError(f"Unknown defense_method: {agg_method}. Available: {list(aggregator_factory.keys())}")

        # --- Proxy Dataset Setup (reused for reputation assessment) --
        # The aggregator will decide if it needs this.
        if agg_method == 'fedrep':
            self.logger.info("Initializing proxy dataset for FedRep reputation assessment.")
            proxy_samples = self.config.get('proxy_dataset_size', 500)
            proxy_data = self.dataset.create_proxy_dataset(proxy_samples)
            
            users = torch.LongTensor([d[0] for d in proxy_data])
            items = torch.LongTensor([d[1] for d in proxy_data])
            ratings = torch.FloatTensor([d[2] for d in proxy_data])
            
            proxy_torch_dataset = TensorDataset(users, items, ratings)
            # Use smaller batches for more granular evaluation of single updates
            self.proxy_dataloader = DataLoader(proxy_torch_dataset, batch_size=self.config.get('proxy_batch_size', 128))


    def select_clients(self, all_client_ids, num_clients_to_select):
        self.logger.info(f"Selecting {num_clients_to_select} clients from a pool of {len(all_client_ids)}.")
        selected_clients = np.random.choice(
            all_client_ids,
            size=num_clients_to_select,
            replace=False
        )
        self.logger.info(f"Selected {len(selected_clients)} client IDs: {selected_clients[:5]}...")
        return selected_clients.tolist()


    def aggregate_updates(self, client_updates, selected_clients):
        # This method remains unchanged, it just calls the aggregator
        return self.aggregator.aggregate(client_updates, selected_clients)

    def update_model(self, aggregated_update):
        # This method also remains unchanged. We are encapsulating all logic
        # into the new aggregator, keeping the server clean.
        if aggregated_update is None:
            self.logger.info("Skipping model update as aggregated_update is None.")
            return

        current_state = self.model.state_dict()
        for key in current_state:
            current_state[key] = current_state[key] + aggregated_update[key].to(self.device)
        self.model.load_state_dict(current_state)
        self.logger.info("Global model updated with aggregated weights.")
        
    def evaluate(self):
        """
        Evaluates the global model on the test dataset.
        """
        self.model.eval()
        self.model.to(self.device)

        # Get the test data from the dataset object
        test_data = self.dataset.test_ground_truth
        k = self.config.get('eval_k', 10)
        
        all_recalls = []
        all_ndcgs = []

        all_user_ids = list(test_data.keys())
        
        with torch.no_grad():
            for user_id in all_user_ids:
                # Get all items for prediction
                item_ids = torch.arange(self.dataset.num_items).to(self.device)
                user_ids_tensor = torch.LongTensor([user_id] * self.dataset.num_items).to(self.device)
                
                # Get predictions for all items for the current user
                predictions = self.model(user_ids_tensor, item_ids)
                
                # Get ground truth items
                ground_truth_items = test_data[user_id]
                if not ground_truth_items:
                    continue # Skip users with no items in the test set

                # Exclude items the user has already interacted with in the training set
                # This is crucial for fair evaluation
                train_items = self.dataset.train_data.get(user_id, [])
                if train_items:
                    predictions[train_items] = -np.inf # Set their score to be very low

                # Get top K recommendations
                _, top_k_indices = torch.topk(predictions, k)
                top_k_items = top_k_indices.cpu().tolist()

                # Calculate Recall and NDCG for this user
                hits = set(top_k_items) & set(ground_truth_items)
                
                # Recall
                recall = len(hits) / len(ground_truth_items) if ground_truth_items else 0
                all_recalls.append(recall)
                
                # NDCG
                idcg_len = min(len(ground_truth_items), k)
                idcg = np.sum([1.0 / np.log2(i + 2) for i in range(idcg_len)])
                
                dcg = 0.0
                for i, item in enumerate(top_k_items):
                    if item in hits:
                        dcg += 1.0 / np.log2(i + 2)
                
                ndcg = dcg / idcg if idcg > 0 else 0
                all_ndcgs.append(ndcg)

        avg_recall = np.mean(all_recalls) if all_recalls else 0.0
        avg_ndcg = np.mean(all_ndcgs) if all_ndcgs else 0.0

        return avg_recall, avg_ndcg

