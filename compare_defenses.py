# compare_defenses.py
# -*- coding: utf-8 -*-

import torch
import os
import time
from copy import deepcopy

from config import CONFIG
from src.data_loader import load_and_preprocess_movielens, create_federated_data
from src.server import Server

def run_experiment_with_defense(defense_method, base_config, processed_df, num_users, num_items, train_data, test_data):
    # 运行单个防御方法的实验
    
    # 创建该防御方法专用的配置
    config = deepcopy(base_config)
    config['defense_method'] = defense_method
    
    # 创建日志文件
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    timestamp = time.strftime('%Y%m%d-%H%M%S')
    log_file_path = os.path.join(log_dir, f"compare_{defense_method}_{timestamp}.log")
    
    def log_message(message):
        print(message)
        with open(log_file_path, 'a', encoding='utf-8') as f:
            f.write(message + '\n')

    log_message("="*80)
    log_message(f"EXPERIMENT: {defense_method.upper()}")
    log_message("="*80)
    log_message(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log_message("\nConfiguration:")
    for key, value in config.items():
        log_message(f"  {key}: {value}")
    log_message("-" * 80)

    # 初始化服务器
    server = Server(
        config=config,
        num_users=num_users,
        num_items=num_items,
        train_data=train_data,
        test_data=test_data
    )

    # 存储实验结果
    results = {
        'rounds': [],
        'recall': [],
        'ndcg': [],
        'attack_success': []  # 如果实现了攻击成功率评估
    }

    # 联邦学习主循环
    log_message("\n--- Starting Federated Training ---")
    start_time = time.time()
    
    for round_num in range(1, config['num_rounds'] + 1):
        # 训练
        round_start_time = time.time()
        server.train_round(round_num)
        round_time = time.time() - round_start_time
        
        # 评估
        if round_num % config['eval_every'] == 0:
            eval_start_time = time.time()
            recall, ndcg = server.evaluate()
            eval_time = time.time() - eval_start_time
            
            # 记录结果
            results['rounds'].append(round_num)
            results['recall'].append(recall)
            results['ndcg'].append(ndcg)
            
            log_message(f"Round {round_num}/{config['num_rounds']} | "
                       f"Recall@{config['eval_k']}: {recall:.4f}, "
                       f"NDCG@{config['eval_k']}: {ndcg:.4f} | "
                       f"Round Time: {round_time:.2f}s, Eval Time: {eval_time:.2f}s")
            
            # 如果实现了攻击成功率评估，可以添加
            # attack_success = server.evaluate_attack_success()  
            # results['attack_success'].append(attack_success)
            # log_message(f"Attack Success Rate: {attack_success:.4f}")

    total_time = time.time() - start_time
    log_message("--- Federated Training Finished ---")
    log_message(f"Total Training Time: {total_time:.2f} seconds")
    
    # 记录最终结果摘要
    if results['recall']:
        final_recall = results['recall'][-1]
        final_ndcg = results['ndcg'][-1]
        max_recall = max(results['recall'])
        max_ndcg = max(results['ndcg'])
        
        log_message("\n" + "="*50)
        log_message("FINAL RESULTS SUMMARY")
        log_message("="*50)
        log_message(f"Defense Method: {defense_method.upper()}")
        log_message(f"Final Recall@{config['eval_k']}: {final_recall:.4f}")
        log_message(f"Final NDCG@{config['eval_k']}: {final_ndcg:.4f}")
        log_message(f"Best Recall@{config['eval_k']}: {max_recall:.4f}")
        log_message(f"Best NDCG@{config['eval_k']}: {max_ndcg:.4f}")
        log_message(f"Attack Settings: {config['malicious_fraction']*100}% malicious clients, "
                   f"amplification factor: {config['attack_amplification_factor']}")
        log_message("="*50)
    
    # 返回结果和日志文件路径
    return results, log_file_path

def main():
    print("=" * 80)
    print("FEDERATED RECOMMENDATION DEFENSE COMPARISON EXPERIMENT")
    print("=" * 80)
    print(f"Experiment started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 加载数据（只加载一次）
    print("\n--- Loading Data ---")
    processed_df, num_users, num_items = load_and_preprocess_movielens(CONFIG['min_interactions'])
    train_data, test_data = create_federated_data(processed_df)
    
    # 定义要比较的防御方法
    defense_methods = ['fedavg', 'fedcsr']
    
    # 存储所有实验结果
    all_results = {}
    all_log_files = {}
    
    # 运行每种防御方法的实验
    for i, defense_method in enumerate(defense_methods):
        print(f"\n{'='*60}")
        print(f"RUNNING EXPERIMENT {i+1}/{len(defense_methods)}: {defense_method.upper()}")
        print(f"{'='*60}")
        
        try:
            results, log_file = run_experiment_with_defense(
                defense_method, CONFIG, processed_df, num_users, num_items, train_data, test_data
            )
            all_results[defense_method] = results
            all_log_files[defense_method] = log_file
            
            print(f"? {defense_method.upper()} experiment completed successfully")
            print(f"  Log file: {log_file}")
            
        except Exception as e:
            print(f"? {defense_method.upper()} experiment failed: {str(e)}")
            continue
    
    # 创建对比总结日志
    timestamp = time.strftime('%Y%m%d-%H%M%S')
    summary_log_path = os.path.join("logs", f"comparison_summary_{timestamp}.log")
    
    with open(summary_log_path, 'w', encoding='utf-8') as f:
        f.write("FEDERATED RECOMMENDATION DEFENSE COMPARISON SUMMARY\n")
        f.write("=" * 80 + "\n")
        f.write(f"Experiment completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Attack settings: {CONFIG['malicious_fraction']*100}% malicious clients, ")
        f.write(f"amplification factor: {CONFIG['attack_amplification_factor']}\n\n")
        
        # 对比结果
        f.write("PERFORMANCE COMPARISON:\n")
        f.write("-" * 50 + "\n")
        
        for defense, results in all_results.items():
            if results['recall']:
                final_recall = results['recall'][-1]
                final_ndcg = results['ndcg'][-1]
                max_recall = max(results['recall'])
                max_ndcg = max(results['ndcg'])
                
                f.write(f"\n{defense.upper()}:\n")
                f.write(f"  Final Recall@{CONFIG['eval_k']}: {final_recall:.4f}\n")
                f.write(f"  Final NDCG@{CONFIG['eval_k']}: {final_ndcg:.4f}\n")
                f.write(f"  Best Recall@{CONFIG['eval_k']}: {max_recall:.4f}\n")
                f.write(f"  Best NDCG@{CONFIG['eval_k']}: {max_ndcg:.4f}\n")
                f.write(f"  Log file: {all_log_files[defense]}\n")
        
        # 如果有多个方法的结果，进行对比
        if len(all_results) >= 2:
            f.write(f"\nCOMPARISON ANALYSIS:\n")
            f.write("-" * 30 + "\n")
            
            methods = list(all_results.keys())
            if len(methods) == 2:
                method1, method2 = methods
                if all_results[method1]['recall'] and all_results[method2]['recall']:
                    recall1 = all_results[method1]['recall'][-1]
                    recall2 = all_results[method2]['recall'][-1]
                    ndcg1 = all_results[method1]['ndcg'][-1]
                    ndcg2 = all_results[method2]['ndcg'][-1]
                    
                    f.write(f"Recall improvement ({method2} vs {method1}): ")
                    f.write(f"{((recall2 - recall1) / recall1 * 100):+.2f}%\n")
                    f.write(f"NDCG improvement ({method2} vs {method1}): ")
                    f.write(f"{((ndcg2 - ndcg1) / ndcg1 * 100):+.2f}%\n")
    
    print(f"\n{'='*80}")
    print("COMPARISON EXPERIMENT COMPLETED")
    print(f"{'='*80}")
    print(f"Summary log: {summary_log_path}")
    print("Individual experiment logs:")
    for defense, log_file in all_log_files.items():
        print(f"  {defense.upper()}: {log_file}")

if __name__ == '__main__':
    main()
