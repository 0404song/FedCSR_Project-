# src/dataset.py (Final, Class-based Version)
import os
import torch
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class RecSysDataset:
   
    def __init__(self, data_path, min_interactions=5, test_size=0.2, random_state=42):
        self.data_path = Path(data_path)
        self.min_interactions = min_interactions
        self.test_size = test_size
        self.random_state = random_state

        # Run the full pipeline
        self.df, self.num_users, self.num_items, self.user_map, self.item_map = self._load_and_preprocess()
        self.train_data, self.test_data = self._create_federated_data()
        self.test_ground_truth = self._get_test_ground_truth()

    def _load_and_preprocess(self):
        #"""Loads, filters, and remaps the MovieLens dataset."""
        dataset_file = self.data_path / 'ratings.dat'
        if not dataset_file.exists():
            raise FileNotFoundError(f"Dataset file not found at: {dataset_file}")

        df = pd.read_csv(dataset_file, sep='::', header=None,
                         names=['RawUserID', 'RawMovieID', 'Rating', 'Timestamp'], engine='python')
        
        # Keep positive feedback
        df = df[df['Rating'] >= 4].copy()

        # 5-core filtering
        while True:
            user_counts = df['RawUserID'].value_counts()
            item_counts = df['RawMovieID'].value_counts()
            if user_counts.min() >= self.min_interactions and item_counts.min() >= self.min_interactions:
                break
            
            inactive_users = user_counts[user_counts < self.min_interactions].index
            inactive_items = item_counts[item_counts < self.min_interactions].index
            
            df = df[~df['RawUserID'].isin(inactive_users)]
            df = df[~df['RawMovieID'].isin(inactive_items)]

        # Remap IDs
        unique_users = df['RawUserID'].unique()
        unique_items = df['RawMovieID'].unique()
        user_map = {id: i for i, id in enumerate(unique_users)}
        item_map = {id: i for i, id in enumerate(unique_items)}

        df['user_id'] = df['RawUserID'].map(user_map)
        df['item_id'] = df['RawMovieID'].map(item_map)

        num_users = len(unique_users)
        num_items = len(unique_items)
        
        return df[['user_id', 'item_id']], num_users, num_items, user_map, item_map

    def _create_federated_data(self):
        #"""Splits data into federated train/test sets for each client."""
        train_data = {}
        test_data = {}
        
        user_groups = self.df.groupby('user_id')
        for user_id, items_df in user_groups:
            if len(items_df) < 5:  # Not enough items to create a test set
                train_data[user_id] = items_df['item_id'].tolist()
                test_data[user_id] = []
                continue
            
            train_items, test_items = train_test_split(items_df, test_size=self.test_size, random_state=self.random_state)
            train_data[user_id] = train_items['item_id'].tolist()
            test_data[user_id] = test_items['item_id'].tolist()
            
        return train_data, test_data

    def get_client_data(self, client_id):
        #"""Returns the training data for a specific client."""
        return self.train_data.get(client_id, [])

    def _get_test_ground_truth(self):
        #"""Prepares the test data in a format suitable for evaluation."""
        return {k: v for k, v in self.test_data.items() if v} # Exclude users with no test items

# This class is needed for local training on each client
class LocalDataset(Dataset):
    def __init__(self, user_id, item_list):
        self.user_ids = torch.LongTensor([user_id] * len(item_list))
        self.item_ids = torch.LongTensor(item_list)

    def __len__(self):
        return len(self.item_ids)

    def __getitem__(self, idx):
        return self.user_ids[idx], self.item_ids[idx]

if __name__ == '__main__':
    # For testing purposes
    print("Testing RecSysDataset...")
    # NOTE: You need to adjust the path to your data directory
    project_root = Path(__file__).parent.parent 
    dataset = RecSysDataset(data_path=project_root / 'data' / 'ml-1m')
    
    print(f"Number of users: {dataset.num_users}")
    print(f"Number of items: {dataset.num_items}")
    
    first_client_id = 0
    client_train_data = dataset.get_client_data(first_client_id)
    print(f"Train items for client {first_client_id} ({len(client_train_data)} items): {client_train_data[:10]}...")
    
    client_test_data = dataset.test_data.get(first_client_id, [])
    print(f"Test items for client {first_client_id} ({len(client_test_data)} items): {client_test_data[:10]}...")
