# src/data_loader.py (健壮路径版)

import pandas as pd
from sklearn.model_selection import train_test_split
import os
from pathlib import Path  # 导入新的模块 Path


def load_and_preprocess_movielens(min_interactions=5):
    """
    加载并预处理MovieLens-1M数据集。
    此版本使用绝对路径定位，更健壮。

    Args:
        min_interactions (int): 用户和物品的最小交互次数阈值。

    Returns:
        tuple: 包含以下元素的元组：
            - df (pd.DataFrame): 处理后的数据，包含 'user_id', 'item_id' 列。
            - num_users (int): 唯一用户数量。
            - num_items (int): 唯一物品数量。
    """
    print("--- Loading and Preprocessing Data ---")

    # --- 核心修改部分：自动计算绝对路径 ---
    # Path(__file__) 获取当前脚本文件 data_loader.py 的路径
    # .parent 获取该文件的父目录 (即 src 文件夹)
    # .parent.parent 获取 src 的父目录 (即项目根目录 FedCSR_Project)
    project_root = Path(__file__).parent.parent
    dataset_path = project_root / 'data' / 'ml-1m' / 'ratings.dat'
    # ----------------------------------------

    # 检查文件是否存在
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found at: {dataset_path}\n"
                                f"Please make sure the 'ml-1m' folder is in the '{project_root / 'data'}' directory.")

    # 1. 读取数据
    df = pd.read_csv(dataset_path, sep='::', header=None,
                     names=['RawUserID', 'RawMovieID', 'Rating', 'Timestamp'], engine='python')
    print(f"Initial data loaded: {len(df)} ratings.")

    # 2. 转换为隐式反馈 (评分>=4视为正反馈)
    df = df[df['Rating'] >= 4].copy()
    print(f"Data after keeping positive feedback (rating >= 4): {len(df)} interactions.")

    # 3. 过滤低活用户和物品 (5-core filtering)
    while True:
        user_counts = df['RawUserID'].value_counts()
        item_counts = df['RawMovieID'].value_counts()

        inactive_users = user_counts[user_counts < min_interactions].index
        inactive_items = item_counts[item_counts < min_interactions].index

        if len(inactive_users) == 0 and len(inactive_items) == 0:
            break

        df = df[~df['RawUserID'].isin(inactive_users)]
        df = df[~df['RawMovieID'].isin(inactive_items)]
        print(f"Filtering... Removed {len(inactive_users)} inactive users and {len(inactive_items)} inactive items.")

    print(f"Data after filtering: {len(df)} interactions.")

    # 4. ID重映射 (从0开始的连续整数)
    unique_users = df['RawUserID'].unique()
    unique_items = df['RawMovieID'].unique()

    user_map = {id: i for i, id in enumerate(unique_users)}
    item_map = {id: i for i, id in enumerate(unique_items)}

    df['user_id'] = df['RawUserID'].map(user_map)
    df['item_id'] = df['RawMovieID'].map(item_map)

    num_users = len(user_map)
    num_items = len(item_map)

    print(f"Final processed data: {num_users} users, {num_items} items.")
    print("--- Data Loading and Preprocessing Finished ---\n")

    return df[['user_id', 'item_id']], num_users, num_items


def create_federated_data(df):
    """
    将处理好的数据划分为联邦化格式 (每个用户一个客户端)。

    Args:
        df (pd.DataFrame): 必须包含 'user_id' 和 'item_id' 列。

    Returns:
        tuple: 包含以下元素的元组：
            - train_data (dict): key为客户端ID, value为其本地训练物品列表。
            - test_data (dict): key为客户端ID, value为其本地测试物品列表。
    """
    print("--- Creating Federated Data Partition ---")

    train_data = {}
    test_data = {}

    user_groups = df.groupby('user_id')

    for user_id, items_df in user_groups:
        if len(items_df) < 5:
            train_data[user_id] = items_df['item_id'].tolist()
            test_data[user_id] = []
            continue

        train_items, test_items = train_test_split(items_df, test_size=0.2, random_state=42)
        train_data[user_id] = train_items['item_id'].tolist()
        test_data[user_id] = test_items['item_id'].tolist()

    print(f"Federated data created for {len(train_data)} clients.")
    print("--- Federated Data Partition Finished ---\n")
    return train_data, test_data


if __name__ == '__main__':
    processed_df, num_users, num_items = load_and_preprocess_movielens()

    print("Sample of processed data:")
    print(processed_df.head())
    print("\n")

    train_data, test_data = create_federated_data(processed_df)

    first_client_id = list(train_data.keys())[0]
    print(f"Data for Client ID {first_client_id}:")
    print(f"  Training items ({len(train_data[first_client_id])} items): {train_data[first_client_id][:10]}...")
    print(f"  Test items ({len(test_data[first_client_id])} items): {test_data[first_client_id][:10]}...")
