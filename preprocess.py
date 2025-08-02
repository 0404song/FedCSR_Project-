# preprocess.py
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os

# --- 配置参数 ---
DATA_DIR = './data/movielens/'
OUTPUT_DIR = './data/movielens/'
MIN_USER_COUNT = 5  # 每位用户至少有5次评分
MIN_ITEM_COUNT = 5  # 每个物品至少被5个用户评论过
TEST_SIZE_PER_USER = 1  # 为每个用户保留1个物品用于测试

print("--- 开始预处理MovieLens-1M数据集 ---")

# 1. 加载数据
print("1/5: 正在加载 ratings.dat...")
ratings_df = pd.read_csv(
    os.path.join(DATA_DIR, 'ratings.dat'),
    sep='::',
    header=None,
    names=['user_id', 'item_id', 'rating', 'timestamp'],
    engine='python'
)

# 2. 过滤低活用户和物品
print("2/5: 正在过滤低活用户和物品...")
while True:
    user_counts = ratings_df['user_id'].value_counts()
    item_counts = ratings_df['item_id'].value_counts()
    
    # 筛选出活跃的用户和物品
    active_users = user_counts[user_counts >= MIN_USER_COUNT].index
    active_items = item_counts[item_counts >= MIN_ITEM_COUNT].index
    
    # 过滤DataFrame
    before_count = len(ratings_df)
    ratings_df = ratings_df[(ratings_df['user_id'].isin(active_users)) & (ratings_df['item_id'].isin(active_items))]
    after_count = len(ratings_df)
    
    print(f"    本轮过滤后剩余: {after_count} 条记录")
    
    # 如果没有记录被过滤，则停止循环
    if before_count == after_count:
        break

# 3. 重新映射用户ID和物品ID，使其从0开始连续
print("3/5: 正在重新映射ID...")
unique_users = ratings_df['user_id'].unique()
unique_items = ratings_df['item_id'].unique()

user_map = {old_id: new_id for new_id, old_id in enumerate(unique_users)}
item_map = {old_id: new_id for new_id, old_id in enumerate(unique_items)}

ratings_df['user_id'] = ratings_df['user_id'].map(user_map)
ratings_df['item_id'] = ratings_df['item_id'].map(item_map)

print(f"    处理后用户数: {len(user_map)}")
print(f"    处理后物品数: {len(item_map)}")

# 4. 排序并拆分训练/测试集
print("4/5: 正在拆分训练集和测试集...")
# 按用户和时间排序，确保测试集是用户最近的交互
ratings_df = ratings_df.sort_values(by=['user_id', 'timestamp'])

train_data = {}
test_data = {}

# 使用groupby来高效处理每个用户
for user_id, group in ratings_df.groupby('user_id'):
    items = group['item_id'].tolist()
    if len(items) > TEST_SIZE_PER_USER:
        train_data[user_id] = items[:-TEST_SIZE_PER_USER]
        test_data[user_id] = items[-TEST_SIZE_PER_USER:]
    else:
        # 如果物品数量不足，全部作为训练数据
        train_data[user_id] = items

# 5. 保存为指定格式的文件
print("5/5: 正在保存 train.txt 和 test.txt...")

def save_to_file(data_dict, file_path):
    with open(file_path, 'w') as f:
        for user_id in sorted(data_dict.keys()):
            items_str = ' '.join(map(str, data_dict[user_id]))
            f.write(f'{user_id} {items_str}\n')

save_to_file(train_data, os.path.join(OUTPUT_DIR, 'train.txt'))
save_to_file(test_data, os.path.join(OUTPUT_DIR, 'test.txt'))

print("--- 预处理完成！---")
print(f"文件已保存至: {OUTPUT_DIR}")
