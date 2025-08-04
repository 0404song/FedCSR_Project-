# src/client.py (重构版，支持多种攻击)
# -*- coding: utf-8 -*-

import torch
import torch.optim as optim
import random
from copy import deepcopy
from .model import bpr_loss

class Client:
    def __init__(self, client_id, local_train_data, all_item_ids, num_items, device, is_malicious=False, config=None):
        self.id = client_id
        self.local_train_data = set(local_train_data)
        self.all_item_ids = all_item_ids
        self.num_items = num_items
        self.device = device
        self.is_malicious = is_malicious
        self.config = config

    def _benign_sampling(self, batch_size):
        """良性客户端的采样方式"""
        if not self.local_train_data:
            return [], []
        
        effective_batch_size = min(batch_size, len(self.local_train_data))
        pos_items = random.sample(list(self.local_train_data), effective_batch_size)
        neg_items = []
        for _ in pos_items:
            neg_item = random.choice(list(self.all_item_ids))
            while neg_item in self.local_train_data:
                neg_item = random.choice(list(self.all_item_ids))
            neg_items.append(neg_item)
        return pos_items, neg_items

    def _malicious_poisoning(self, batch_size):
        """恶意客户端的攻击方式：推广目标物品"""
        target_item = self.config['target_item_id']
        pos_items = [target_item] * batch_size
        
        neg_items = []
        for _ in pos_items:
            neg_item = random.choice(list(self.all_item_ids))
            while neg_item == target_item:
                neg_item = random.choice(list(self.all_item_ids))
            neg_items.append(neg_item)
        return pos_items, neg_items

    def train(self, model_template, local_epochs, batch_size, lr):
        local_model = deepcopy(model_template).to(self.device)
        optimizer = optim.Adam(local_model.parameters(), lr=lr)
        local_model.train()
        global_params_cpu = {k: v.cpu() for k, v in model_template.state_dict().items()}

        # 恶意客户端执行符号翻转攻击时，先按良性方式训练
        attack_logic = self.config.get('attack_logic', 'poison') # 默认为之前的poison攻击
        is_sign_flipping = self.is_malicious and attack_logic == 'sign_flipping'
        
        # 训练循环
        for epoch in range(local_epochs):
            num_batches = len(self.local_train_data) // batch_size if self.local_train_data else 0
            if num_batches == 0: break

            for _ in range(num_batches):
                if self.is_malicious and attack_logic == 'poison':
                    batch_pos_items, batch_neg_items = self._malicious_poisoning(batch_size)
                else: # 良性客户端和执行符号翻转的恶意客户端都使用良性采样
                    batch_pos_items, batch_neg_items = self._benign_sampling(batch_size)

                if not batch_pos_items: continue
                
                user_indices = torch.LongTensor([self.id] * len(batch_pos_items)).to(self.device)
                pos_item_indices = torch.LongTensor(batch_pos_items).to(self.device)
                neg_item_indices = torch.LongTensor(batch_neg_items).to(self.device)

                optimizer.zero_grad()
                pos_scores, neg_scores = local_model(user_indices, pos_item_indices, neg_item_indices)
                loss = bpr_loss(pos_scores, neg_scores)
                
                loss.backward()
                optimizer.step()
        
        # 计算模型更新
        local_params_cpu = {k: v.cpu() for k, v in local_model.state_dict().items()}
        model_update = {key: local_params_cpu[key] - global_params_cpu[key] for key in global_params_cpu.keys()}
            
        # 根据攻击类型，对模型更新进行后处理
        if self.is_malicious:
            # 符号翻转攻击
            if attack_logic == 'sign_flipping':
                factor = self.config.get("attack_amplification_factor", 5.0) # 默认为-5
                print(f"Client {self.id}: Applying sign-flipping attack with factor -{factor}")
                for key in model_update:
                    model_update[key] *= -factor
            
            # 原本的梯度放大攻击 (配合poison)
            elif attack_logic == 'poison':
                factor = self.config.get("attack_amplification_factor", 1.0)
                if factor > 1.0:
                    print(f"Client {self.id}: Amplifying poisoning attack by {factor}x")
                    for key in model_update:
                        model_update[key] *= factor

        return model_update

