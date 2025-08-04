# config.py (最终决战版)

import torch

CONFIG = {
    # --- 数据和模型配置 ---
    "data_path": "./data/ml-1m/",
    "min_interactions": 5,
    "embedding_dim": 32,
    # "num_clients": 6040, # 这个参数通常由数据加载器动态确定，建议注释掉或移除

    # --- 联邦学习配置 ---
    "num_rounds": 50,
    "client_fraction": 0.1,  # 每轮选择10%的客户端
    "local_epochs": 5,
    "batch_size": 32,
    "learning_rate": 0.01,
    "device": "cuda" if torch.cuda.is_available() else "cpu",

    # =======================================================================
    #               核心实验控制区 (重点关注这里)
    # =======================================================================

    # --- 1. 防御方法选择 ---
    # 第一次运行设置为 'fedcsr'，第二次运行设置为 'fedavg'
    "defense_method": "fedcsr",  #  <--- 先跑 'fedcsr'

    # --- 2. 攻击配置 (使用新的毁灭性攻击) ---
    "malicious_fraction": 0.3,          # 30%的恶意客户端，比例更高，更容易摧毁FedAvg
    "attack_type": "model_poisoning",   # 攻击类型名称，可以任意取
    
    # [关键] 这个参数会告诉 client.py 使用哪种攻击逻辑
    "attack_logic": "sign_flipping",    # <-- 必须是 'sign_flipping' 来激活新攻击
    
    # [关键] 符号翻转攻击的强度因子 (乘以 -X)
    "attack_amplification_factor": 5.0, # <-- 5.0 是一个很强的负向放大

    # --- 3. FedCSR 独有配置 (为新防御逻辑调优) ---
    # 这些参数将在 defense_method=='fedcsr' 时被你的新 server.py 代码使用
    
    "num_clusters": 3,
    "kmeans_batch_size": 20, # 这个参数在我的最新建议中没有用到，但可以保留
    
    # [关键] 对应 _norm_based_filtering 中的乘数，4.0 是一个比较合理的值
    "gamma_norm_filter": 3.0,
    
    # [关键] 对应 _coordinate_wise_filtering 中的鲁棒z-score阈值
    "beta_coord_filter": 10.0,
    
    # [关键] 对应 aggregate 方法中的最小簇大小保护
    "min_cluster_size_ratio": 0.05, # 保护成员数占本轮参与者5%以上的簇

    "lambda_reputation": 0.95,  # 信誉更新的遗忘因子 λ，0.95让声誉变化更平滑

    # =======================================================================
    #                       其他和评估配置
    # =======================================================================
    
    # "target_item_id" 在 sign_flipping 攻击中不会被使用，但可以保留
    "target_item_id": 42,
    
    "eval_k": 10,
    "eval_every": 5, # 每5轮评估一次
}
