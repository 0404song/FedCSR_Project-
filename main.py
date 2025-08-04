# -*- coding: utf-8 -*-

# main.py (重构版)

import torch
import os
import time

# 将src目录添加到系统路径，以便导入模块
# import sys
# sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from config import CONFIG
from src.data_loader import load_and_preprocess_movielens, create_federated_data # 修改点
from src.server import Server 

def main():
    # --- 1. 设置和加载 ---
    print("Using device:", CONFIG['device'])
    
    # 创建日志文件
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file_path = os.path.join(log_dir, f"{CONFIG['defense_method']}_{time.strftime('%Y%m%d-%H%M%S')}.log")
    
    def log_message(message):
        print(message)
        with open(log_file_path, 'a') as f:
            f.write(message + '\n')

    log_message("Configuration:\n" + str(CONFIG))


    # --- 2. 加载数据 ---
    # 注意：你的 data_loader.py 使用了 Path(__file__)，
    # 所以请确保你的目录结构是:
    # FedCSR_Project/
    #  |- main.py
    #  |- config.py
    #  |- src/
    #  |  |- data_loader.py
    #  |  |- ...
    #  |- data/
    #  |  |- ml-1m/
    #  |  |  - ratings.dat
    processed_df, num_users, num_items = load_and_preprocess_movielens(CONFIG['min_interactions'])
    train_data, test_data = create_federated_data(processed_df)


    # --- 3. 初始化服务器 ---
    # 服务器内部会根据config创建客户端
    server = Server(
        config=CONFIG,
        num_users=num_users,
        num_items=num_items,
        train_data=train_data,
        test_data=test_data
    )

    # --- 4. 联邦学习主循环 ---
    log_message("--- Starting Federated Training ---")
    for round_num in range(1, CONFIG['num_rounds'] + 1):
        # 训练
        server.train_round(round_num)

        # 评估
        if round_num % CONFIG['eval_every'] == 0:
            recall, ndcg = server.evaluate()
            log_message(f"Round {round_num}/{CONFIG['num_rounds']} | Recall@{CONFIG['eval_k']}: {recall:.4f}, NDCG@{CONFIG['eval_k']}: {ndcg:.4f}")

    log_message("--- Federated Training Finished ---")


if __name__ == '__main__':
    # 确保你的 sklearn, pandas, torch, tqdm 已经安装
    # pip install scikit-learn pandas torch tqdm
    
    # 请确认你的文件目录结构正确
    # 如果 data_loader.py 报错找不到文件，请检查 main.py 所在的位置
    # main.py 应该在项目的根目录
    
    main()
