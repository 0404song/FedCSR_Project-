# src/aggregators.py

import torch
import numpy as np
import logging
from sklearn.cluster import MiniBatchKMeans

class AggregatorBase:
    """A base class for all aggregation strategies."""
    def __init__(self, config, logger=None):
        self.config = config
        self.logger = logger if logger else logging.getLogger(self.__class__.__name__)

    def aggregate(self, client_updates, selected_clients_indices=None):
        """The main aggregation method to be implemented by subclasses."""
        raise NotImplementedError

class FedAvgAggregator(AggregatorBase):
    """Standard Federated Averaging."""
    def aggregate(self, client_updates, selected_clients_indices=None):
        if not client_updates:
            self.logger.info("No client updates to aggregate.")
            return None
        
        agg_update = {key: torch.zeros_like(param) for key, param in client_updates[0].items()}
        
        for update_dict in client_updates:
            for key in agg_update:
                agg_update[key] += update_dict[key].to(agg_update[key].device)
        
        num_updates = len(client_updates)
        if num_updates > 0:
            for key in agg_update:
                agg_update[key] /= num_updates
        
        self.logger.info(f"--- [FedAvg Aggregation] Averaged {num_updates} client updates. ---")
        return agg_update

class FedNormAggregator(AggregatorBase):
    """Federated Averaging with Norm Clipping."""
    def __init__(self, config, logger=None):
        super().__init__(config, logger)
        self.clipping_percentile = self.config.get('norm_clipping_percentile', 90.0)

    def aggregate(self, client_updates, selected_clients_indices=None):
        if not client_updates:
            return None

        # 1. Flatten all updates and calculate their L2 norms
        flat_updates = []
        for update_dict in client_updates:
            flat_vector = torch.cat([p.view(-1) for p in update_dict.values()])
            flat_updates.append(flat_vector)
        
        norms = [torch.norm(v).item() for v in flat_updates]
        
        # 2. Determine the clipping threshold
        if not norms:
             return None
        threshold = np.percentile(norms, self.clipping_percentile)
        self.logger.info(f"--- [FedNorm Aggregation] Clipping threshold (p{self.clipping_percentile:.1f}): {threshold:.4f} ---")

        # 3. Clip updates and aggregate
        agg_update = {key: torch.zeros_like(param) for key, param in client_updates[0].items()}
        num_clipped = 0
        
        for i, update_dict in enumerate(client_updates):
            norm = norms[i]
            scale = 1.0
            if norm > threshold and threshold > 0: # Avoid clipping if threshold is 0
                scale = threshold / (norm + 1e-6)
                num_clipped += 1
            
            for key in agg_update:
                agg_update[key] += update_dict[key] * scale
        
        num_updates = len(client_updates)
        if num_updates > 0:
            for key in agg_update:
                agg_update[key] /= num_updates

        self.logger.info(f"Aggregation complete. Clipped {num_clipped}/{num_updates} updates.")
        return agg_update

class FedCSRHardFilter(AggregatorBase):
    """FedCSR v3: Hard-Filtering based on clustering."""
    def __init__(self, config, logger=None):
        super().__init__(config, logger)
        self.kmeans = MiniBatchKMeans(
            n_clusters=self.config['num_clusters'],
            random_state=42,
            batch_size=self.config['kmeans_batch_size'],
            n_init='auto'
        )

    def _identify_main_cluster(self, kmeans_model, labels, updates):
        unique_labels, counts = np.unique(labels, return_counts=True)
        min_cluster_size = self.config['min_cluster_size_ratio'] * len(updates)
        
        candidate_clusters = [
            label for label, count in zip(unique_labels, counts)
            if count >= min_cluster_size
        ]
        
        if not candidate_clusters:
            return None, 0
        
        cluster_centers = kmeans_model.cluster_centers_
        main_cluster_label = min(candidate_clusters, key=lambda i: np.linalg.norm(cluster_centers[i]))
        main_cluster_size = counts[np.where(unique_labels == main_cluster_label)[0][0]]
        
        return main_cluster_label, main_cluster_size
        
    def _flatten_updates(self, client_updates):
        flat_arrays = []
        for update_dict in client_updates:
            flat_vector = torch.cat([p.view(-1) for p in update_dict.values()])
            flat_arrays.append(flat_vector.cpu().numpy())
        return np.array(flat_arrays)

    def aggregate(self, client_updates, selected_clients_indices=None):
        self.logger.info("--- [FedCSR v3.0: Hard-Filtering Aggregation] ---")
        if not client_updates:
            return None

        flat_updates = self._flatten_updates(client_updates)
        labels = self.kmeans.fit_predict(flat_updates)
        main_cluster_label, _ = self._identify_main_cluster(self.kmeans, labels, flat_updates)
        
        if main_cluster_label is None:
            self.logger.warning("Could not identify a main cluster. Skipping aggregation.")
            return None

        main_cluster_indices = np.where(labels == main_cluster_label)[0]
        
        if len(main_cluster_indices) == 0:
            self.logger.warning("Main cluster is empty. Skipping aggregation.")
            return None
            
        benign_updates = [client_updates[i] for i in main_cluster_indices]
        num_benign = len(benign_updates)
        num_total = len(client_updates)
        self.logger.info(f"Identified main cluster '{main_cluster_label}' with {num_benign}/{num_total} clients.")

        agg_update = {key: torch.zeros_like(param) for key, param in benign_updates[0].items()}
        for update_dict in benign_updates:
            for key in agg_update:
                agg_update[key] += update_dict[key].to(agg_update[key].device)
        
        if num_benign > 0:
            for key in agg_update:
                agg_update[key] /= num_benign

        self.logger.info("Aggregation complete based on the main cluster.")
        return agg_update

# You can add FedCSR v4 here later
# class FedCSRDirectional(AggregatorBase):
#    ...
# Add this new class at the end of src/aggregators.py

class FedCSRDirectional(AggregatorBase):
    """FedCSR v4: Directional Clustering Defense."""
    def __init__(self, config, logger=None):
        super().__init__(config, logger)
        self.kmeans = MiniBatchKMeans(
            n_clusters=self.config.get('num_clusters', 3),
            random_state=42,
            batch_size=self.config.get('kmeans_batch_size', 20),
            n_init='auto'
        )

    def _identify_main_cluster(self, kmeans_model, labels):
        # In directional clustering, the largest cluster is the main one.
        # This is because benign clients, despite having diverse data,
        # should still share a common goal directionally.
        unique_labels, counts = np.unique(labels, return_counts=True)
        if len(counts) == 0:
            return None
        
        main_cluster_label = unique_labels[np.argmax(counts)]
        return main_cluster_label

    def aggregate(self, client_updates, selected_clients_indices=None):
        self.logger.info("--- [FedCSR v4: Directional Clustering Aggregation] ---")
        if not client_updates:
            return None

        # 1. Normalize all updates to get unit vectors (directions)
        directions = []
        for update_dict in client_updates:
            flat_vector = torch.cat([p.view(-1) for p in update_dict.values()])
            norm = torch.norm(flat_vector)
            # Avoid division by zero for zero-updates
            if norm > 1e-6:
                directions.append((flat_vector / norm).cpu().numpy())
            else: # Handle zero updates by appending a zero vector
                directions.append(torch.zeros_like(flat_vector).cpu().numpy())
        
        directions = np.array(directions)

        # 2. Perform K-Means clustering on the directions
        labels = self.kmeans.fit_predict(directions)
        
        # 3. Identify the main cluster (the largest one)
        main_cluster_label = self._identify_main_cluster(self.kmeans, labels)
        
        if main_cluster_label is None:
            self.logger.warning("Could not identify a main cluster. Skipping aggregation.")
            return None

        main_cluster_indices = np.where(labels == main_cluster_label)[0]
        
        # 4. Hard-filter: Aggregate the ORIGINAL updates from the main cluster
        if len(main_cluster_indices) == 0:
            self.logger.warning("Main cluster is empty. Skipping aggregation.")
            return None
            
        # IMPORTANT: We aggregate the original, non-normalized updates
        benign_updates = [client_updates[i] for i in main_cluster_indices]
        num_benign = len(benign_updates)
        num_total = len(client_updates)
        self.logger.info(f"Identified main direction cluster '{main_cluster_label}' with {num_benign}/{num_total} clients.")

        agg_update = {key: torch.zeros_like(param) for key, param in benign_updates[0].items()}
        for update_dict in benign_updates:
            for key in agg_update:
                agg_update[key] += update_dict[key].to(agg_update[key].device)
        
        if num_benign > 0:
            for key in agg_update:
                agg_update[key] /= num_benign

        self.logger.info("Aggregation complete based on the main direction cluster.")
        return agg_update
