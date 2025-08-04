# -*- coding: utf-8 -*-

# src/server.py (重构版)

import torch
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from collections import defaultdict
from tqdm import tqdm
import random
import gc

from .model import MatrixFactorization
from .client import Client
from .metrics import recall_at_k, ndcg_at_k

# ==============================================================================
#  [新] 防御模块：FedCSRServer (你的研究核心)
# ==============================================================================
class FedCSRServer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.client_reputations = defaultdict(lambda: 1.0)

    def aggregate(self, client_updates, selected_clients_indices):
        num_selected = len(selected_clients_indices)
        if num_selected == 0:
            return self._get_zero_update()

        print("\n--- [Ultra-Simple Norm Defense] ---")

        # 1. 数据准备：计算所有更新的范数
        updates_with_meta = []
        for i, update in enumerate(client_updates):
            client_id = selected_clients_indices[i]
            flat_update = self._flatten_update(update)
            norm = torch.norm(flat_update).item()
            updates_with_meta.append({
                'id': client_id,
                'update_dict': update,
                'norm': norm
            })
            print(f"Client {client_id}: norm = {norm:.4f}")

        # 2. 核心防御逻辑：只基于范数进行过滤
        norms = [d['norm'] for d in updates_with_meta]
        if not norms:
            return self._get_zero_update()

        # 使用中位数作为良性范数的基准，非常鲁棒
        median_norm = np.median(norms)
        # 我们的gamma现在直接作为中位数的乘数，简单直接！
        threshold = median_norm * self.config['gamma_norm_filter']
        
        print(f"Median norm: {median_norm:.4f}, Threshold: {threshold:.4f}")

        # 3. 识别并聚合良性更新
        final_updates = []
        final_reputations = []
        malicious_ids_this_round = set()
        
        for data in updates_with_meta:
            if data['norm'] > threshold or data['norm'] < 1e-6:
                malicious_ids_this_round.add(data['id'])
            else:
                final_updates.append(data['update_dict'])
                final_reputations.append(self.client_reputations[data['id']])

        print(f"Flagged clients ({len(malicious_ids_this_round)}): {list(malicious_ids_this_round)}")
        num_survived = len(final_updates)
        print(f"Clients survived: {num_survived}")
        
        # 4. 更新信誉分
        self._update_reputations(selected_clients_indices, malicious_ids_this_round)

        # 5. 聚合
        if not final_updates:
            print("All clients were filtered. Skipping model update.")
            return self._get_zero_update()

        return self._reputation_weighted_aggregation(final_updates, final_reputations)


    def _flatten_update(self, update_dict):
        return torch.cat([param.view(-1) for param in update_dict.values()])

    def _unflatten_update(self, flat_tensor, model_state_dict):
        new_dict = {}
        start_idx = 0
        for key, param in model_state_dict.items():
            param_shape = param.shape
            num_elements = param.numel()
            new_dict[key] = flat_tensor[start_idx : start_idx + num_elements].reshape(param_shape).clone()
            start_idx += num_elements
        return new_dict
        
    def _get_zero_update(self):
        return {key: torch.zeros_like(param) for key, param in self.model.state_dict().items()}

    def _cluster_clients(self, flat_updates):
        kmeans = MiniBatchKMeans(
            n_clusters=self.config['num_clusters'], random_state=0,
            batch_size=self.config['kmeans_batch_size'], n_init='auto'
        )
        updates_np = torch.stack(flat_updates).cpu().numpy()
        return kmeans.fit_predict(updates_np)

    def _norm_based_filtering(self, cluster_updates_flat, cluster_original_ids):
        if not cluster_updates_flat: return [], [], set()
        norms = [torch.norm(update).item() for update in cluster_updates_flat]
        
        # 使用百分位数来设定阈值，更鲁棒
        lower_bound = np.percentile(norms, 10)
        upper_bound = np.percentile(norms, 90) * self.config.get('gamma_norm_filter', 4.0) # gamma现在是上界的乘数
    
        passed_indices = []
        flagged_ids = set()
    
        for i, norm in enumerate(norms):
            # 范数过大（可能是攻击）或过小（可能是无效更新）都标记
            if norm > upper_bound or norm < lower_bound * 0.1:
                flagged_ids.add(cluster_original_ids[i])
            else:
                passed_indices.append(i)
        
        if flagged_ids: print(f"FedCSR: Norm filter flagged {len(flagged_ids)} clients.")
        return passed_indices, list(flagged_ids)


    # 替换你的 _coordinate_wise_filtering
    def _coordinate_wise_filtering(self, updates_to_check, ids_to_check):
        if not updates_to_check: return [], set()
        updates_stack = torch.stack(updates_to_check)
        
        # 改为使用中位数和MAD，对攻击更鲁棒
        median_update = torch.median(updates_stack, dim=0).values
        devs = torch.abs(updates_stack - median_update)
        mad_update = torch.median(devs, dim=0).values
        mad_update[mad_update == 0] = 1e-6 # 避免除以0
    
        verified_updates_indices = []
        flagged_ids = set()
    
        for i, update in enumerate(updates_to_check):
            # 计算鲁棒的Z-score
            z_scores = 0.6745 * torch.abs(update - median_update) / mad_update
            
            # 如果任何一个坐标的z-score过高，则标记整个客户端
            if torch.any(z_scores > self.config['beta_coord_filter']):
                flagged_ids.add(ids_to_check[i])
            else:
                verified_updates_indices.append(i)
        
        # 只返回通过了校验的更新
        verified_updates = [updates_to_check[i] for i in verified_updates_indices]
    
        if flagged_ids: print(f"FedCSR: Coord filter flagged {len(flagged_ids)} clients.")
        return verified_updates, flagged_ids


    def _update_reputations(self, selected_clients_indices, malicious_ids_this_round):
        for client_id in selected_clients_indices:
            is_malicious = 1 if client_id in malicious_ids_this_round else 0
            self.client_reputations[client_id] = (
                self.config['lambda_reputation'] * self.client_reputations[client_id] +
                (1 - self.config['lambda_reputation']) * (1 - is_malicious)
            )

    def _reputation_weighted_aggregation(self, updates_dict_list, reputations):
        if not updates_dict_list: return self._get_zero_update()
        total_reputation = sum(reputations)
        if total_reputation == 0: return self._get_zero_update()
        
        agg_update = self._get_zero_update()
        for i, update_dict in enumerate(updates_dict_list):
            weight = reputations[i] / total_reputation
            for key in agg_update:
                agg_update[key] += update_dict[key].to(agg_update[key].device) * weight
        return agg_update

# ==============================================================================
#  [重构后] 的主 Server 类
# ==============================================================================
class Server:
    def __init__(self, config, num_users, num_items, train_data, test_data):
        self.config = config
        self.device = torch.device(config['device'])
        self.global_model = MatrixFactorization(num_users, num_items, config['embedding_dim']).to(self.device)
        self.num_users = num_users
        self.num_items = num_items
        
        all_item_ids = set(range(num_items))
        self.clients = []
        
 #       for client_id in range(num_users):
 #           if client_id in train_data:
 #               client = Client(client_id, train_data[client_id], all_item_ids, num_items, self.device)
 #               self.clients.append(client)
 
        # 确定哪些客户端是恶意的
        num_malicious_clients = int(config['malicious_fraction'] * num_users)
        # 我们选择ID最小的一部分作为恶意客户端，方便观察
        malicious_client_ids = set(range(num_malicious_clients)) 

        for client_id in range(num_users):
            if client_id in train_data:
                # 判断当前客户端是否是恶意的
                is_malicious = (client_id in malicious_client_ids) and (config['attack_type'] != 'none')
                
                # 在创建 Client 对象时传入 is_malicious 标志
                client = Client(
                    client_id=client_id,
                    local_train_data=train_data[client_id],
                    all_item_ids=all_item_ids,
                    num_items=num_items,
                    device=self.device,
                    is_malicious=is_malicious,  # <--- 新增的参数
                    config=config               # <--- 把整个config也传进去
                )
                self.clients.append(client)
        
        if num_malicious_clients > 0:
            print(f"Initialized {num_malicious_clients} malicious clients (IDs 0 to {num_malicious_clients-1}).")     
        self.test_data = test_data

        # --- [核心改造] 根据配置选择聚合器 ---
        if config['defense_method'] == 'fedcsr':
            print("Using FedCSR aggregation strategy.")
            self.aggregator = FedCSRServer(self.global_model, config)
        else: # 'fedavg' or other baselines
            print("Using FedAvg aggregation strategy.")
            # 将聚合函数包装成一个对象，使其接口与FedCSRServer一致
            class FedAvgAggregator:
                def aggregate(self, updates, selected_clients_indices):
                    if not updates: return None
                    keys = updates[0].keys()
                    aggregated_update = {key: torch.zeros_like(updates[0][key]) for key in keys}
                    for key in keys:
                        update_stack = torch.stack([upd[key] for upd in updates], dim=0)
                        aggregated_update[key] = torch.mean(update_stack, dim=0)
                    return aggregated_update
            self.aggregator = FedAvgAggregator()
            
    def select_clients(self):
        fraction = self.config['client_fraction']
        num_selected = max(1, int(len(self.clients) * fraction))
        return random.sample(self.clients, num_selected)

    def aggregate_updates(self, updates, selected_clients_ids):
        aggregated_update = self.aggregator.aggregate(updates, selected_clients_ids)
        
        if aggregated_update:
            with torch.no_grad():
                current_params = self.global_model.state_dict()
                for key in current_params.keys():
                    current_params[key] += aggregated_update[key].to(self.device)
                self.global_model.load_state_dict(current_params)
        
    def train_round(self, num_round):
        print(f"\n--- Training Round {num_round} ---")
        selected_clients = self.select_clients()
        print(f"Selected {len(selected_clients)} clients for this round: {[c.id for c in selected_clients]}")
        
        updates = []
        selected_clients_ids = []
        for client in tqdm(selected_clients, desc="Clients training"):
            update = client.train(
                model_template=self.global_model,
                local_epochs=self.config['local_epochs'],
                batch_size=self.config['batch_size'],
                lr=self.config['learning_rate']
            )
            updates.append(update)
            selected_clients_ids.append(client.id)
            
        print("Aggregating updates...")
        self.aggregate_updates(updates, selected_clients_ids)

        del updates
        gc.collect()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

    def evaluate(self):
        self.global_model.eval()
        k = self.config['eval_k']
        total_recall, total_ndcg, num_test_clients = 0, 0, 0
        all_item_indices = torch.arange(self.num_items).to(self.device)
        
        with torch.no_grad():
            # 只对有测试数据的客户端进行评估
            test_clients = [c for c in self.clients if c.id in self.test_data and self.test_data[c.id]]
            for client in tqdm(test_clients, desc="Evaluating"):
                true_items = self.test_data[client.id]
                num_test_clients += 1
                
                user_index = torch.LongTensor([client.id] * self.num_items).to(self.device)
                user_embedding = self.global_model.user_embeddings(user_index)
                item_embeddings = self.global_model.item_embeddings(all_item_indices)
                scores = torch.sum(user_embedding * item_embeddings, dim=1)
                
                # 评估时不应推荐用户已经交互过的物品
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
