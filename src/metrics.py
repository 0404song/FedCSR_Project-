# src/metrics.py
import torch
import numpy as np

def recall_at_k(y_true_items, y_pred_scores, k=10):
    _, top_k_indices = torch.topk(y_pred_scores, k)
    top_k_items = top_k_indices.cpu().numpy()
    true_items_set = set(y_true_items)
    top_k_items_set = set(top_k_items)
    num_hit = len(true_items_set.intersection(top_k_items_set))
    return num_hit / len(true_items_set) if len(true_items_set) > 0 else 0.0

def ndcg_at_k(y_true_items, y_pred_scores, k=10):
    _, top_k_indices = torch.topk(y_pred_scores, k)
    top_k_items = top_k_indices.cpu().numpy()
    relevance = np.zeros(k)
    true_items_set = set(y_true_items)
    for i, item in enumerate(top_k_items):
        if item in true_items_set:
            relevance[i] = 1
    discounts = np.log2(np.arange(2, k + 2))
    dcg = np.sum(relevance / discounts)
    ideal_relevance = np.zeros(k)
    ideal_relevance[:len(true_items_set)] = 1
    idcg = np.sum(ideal_relevance / discounts)
    return dcg / idcg if idcg > 0 else 0.0
