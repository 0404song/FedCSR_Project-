# main.py (Final, Self-Contained Version for "Decisive Battle")
import os
import sys
import copy
import json
import time
import logging
import torch
import numpy as np
from datetime import datetime

# 将项目根目录添加到Python路径中，以便正确导入模块
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

from config import CONFIG
from src.dataset import RecSysDataset
from src.model import MFModel
from src.client import Client
from src.server import Server

def main():

    # --- 1. 加载配置和设置标准日志系统 ---
    log_level = CONFIG.get("log_level", "INFO").upper()
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    defense_method_str = CONFIG.get('defense_method', 'fedavg')
    log_file_name = f"{defense_method_str}_{log_level}_{datetime.now().strftime('%Y%m%d-%H%M%S')}.log"
    log_file_path = os.path.join(log_dir, log_file_name)
    
    # 使用Python内置的logging模块，这是更标准、更强大的做法
    logging.basicConfig(
        level=log_level,
        format='[%(asctime)s][%(levelname)s] %(message)s',
        datefmt='%H:%M:%S',
        handlers=[
            logging.FileHandler(log_file_path),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger()
    
    logger.info(f"Using device: {CONFIG.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')}")
    logger.info(f"Configuration:\n{json.dumps(CONFIG, indent=4)}")


    # --- 2. 准备数据、模型和客户端 ---
    # 使用我们最新的RecSysDataset
    logger.info("--- Creating Federated Data Partition ---")
    try:
        dataset = RecSysDataset(
            data_path=CONFIG['data_path'],
            min_interactions=CONFIG['min_interactions']
        )
        logger.info(f"Federated data created for {dataset.num_users} clients.")
    except Exception as e:
        logger.error(f"Failed to create dataset: {e}")
        return # Exit if data fails
    logger.info("--- Federated Data Partition Finished ---")

    # 初始化全局模型
    global_model = MFModel(
        num_users=dataset.num_users,
        num_items=dataset.num_items,
        embedding_dim=CONFIG['embedding_dim']
    )
    
 
    # ... (初始化全局模型之后)

    # --- 3. 初始化服务器 ---
    server = Server(
        model=global_model,
        dataset=dataset,
        config=CONFIG
    )
    logger.info(f"Using {CONFIG.get('defense_method', 'fedavg').upper()} aggregation strategy.")


    # --- 4. 开始联邦训练循环 ---
    logger.info("\n--- Starting Federated Training ---")
    start_time = time.time()

    # 获取所有客户端的ID列表
    all_client_ids = list(range(dataset.num_users))
    num_clients_per_round = max(1, int(CONFIG['client_fraction'] * dataset.num_users))
    num_malicious = int(CONFIG['malicious_fraction'] * dataset.num_users)
    if num_malicious > 0:
        logger.info(f"Malicious clients are IDs 0-{num_malicious - 1}.")


    for current_round in range(1, CONFIG['num_rounds'] + 1):
        logger.info(f"\n--- Training Round {current_round}/{CONFIG['num_rounds']} ---")
        
        # 服务器选择客户端ID
         # Modify this call to match the server's method signature
        selected_clients_indices = server.select_clients(
            all_client_ids=all_client_ids, 
            num_clients_to_select=num_clients_per_round
        )
        logger.info(f"Selected {len(selected_clients_indices)} client IDs: {str(selected_clients_indices[:5])}...")

        # [核心优化] 按需创建被选中的客户端对象
        selected_clients = []
        for client_id in selected_clients_indices:
            is_malicious = client_id < num_malicious
            # 为每个选中的客户端创建一个新的、独立的模型副本
            client = Client(
                client_id=client_id,
                model=copy.deepcopy(global_model), # 关键：传递一个干净的模型副本
                dataset=dataset,
                config=CONFIG,
                is_malicious=is_malicious
            )
            selected_clients.append(client)
        
        # 客户端并行（或串行）本地训练
        client_updates = []
        for client in selected_clients:
            update = client.train()
            if update is not None:
                client_updates.append(update)
        
        # [内存清理] 显式删除本轮创建的客户端对象和更新，帮助垃圾回收
        del selected_clients
        if 'torch' in sys.modules and CONFIG.get('device') == 'cuda':
            torch.cuda.empty_cache()

        # 服务器聚合更新
        # (注意：这里的 selected_clients_indices 是ID列表，与FedCSR的reputation系统兼容)
        if client_updates:
            aggregated_update = server.aggregate_updates(client_updates, selected_clients_indices)
            
            # 更新全局模型
            if aggregated_update is not None:
                 server.update_model(aggregated_update)
            else:
                logger.info("Aggregated update is None, skipping model update.")
        else:
            logger.warning("No client updates were generated in this round.")


        # 定期评估
        if current_round % CONFIG['eval_every'] == 0:
            recall, ndcg = server.evaluate()
            logger.info(f"EVALUATION | Round {current_round}/{CONFIG['num_rounds']} | Recall@{CONFIG['eval_k']}: {recall:.4f}, NDCG@{CONFIG['eval_k']}: {ndcg:.4f}")

    end_time = time.time()
    logger.info("\n--- Federated Training Finished ---")
    logger.info(f"Total training time: {(end_time - start_time) / 60:.2f} minutes.")
    logger.info(f"Log file saved to: {log_file_path}")


if __name__ == '__main__':
    main()

