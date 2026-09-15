import json
import numpy as np
from typing import List, Dict
import argparse
import os

def dcg_at_k(relevances: List[float], k: int) -> float:
    """Calculate Discounted Cumulative Gain at k"""
    relevances = np.asarray(relevances)[:k]
    if relevances.size == 0:
        return 0.0
    return np.sum(relevances / np.log2(np.arange(2, relevances.size + 2)))

def ndcg_at_k(ranked_item_ids: List[str], ground_truth_item_ids: List[str], k: int) -> float:
    """Calculate NDCG@K for one user"""
    # Tạo relevance list: 1 nếu item nằm trong ground_truth, 0 otherwise
    relevances = [1.0 if item_id in ground_truth_item_ids else 0.0 for item_id in ranked_item_ids]
    
    dcg = dcg_at_k(relevances, k)
    
    # Ideal DCG: nếu tất cả positive đều nằm ở top
    ideal_relevances = sorted(relevances, reverse=True)
    idcg = dcg_at_k(ideal_relevances, k)
    
    return dcg / idcg if idcg > 0 else 0.0

def recall_at_k(ranked_item_ids: List[str], ground_truth_item_ids: List[str], k: int) -> float:
    """Calculate Recall@K"""
    top_k = set(ranked_item_ids[:k])
    hits = len(top_k & set(ground_truth_item_ids))
    return hits / len(ground_truth_item_ids) if ground_truth_item_ids else 0.0

def analyze_results(input_file: str):
    if not os.path.exists(input_file):
        print(f"Error: File not found: {input_file}")
        return
    
    print(f"Loading results from: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} users\n")
    
    Ks = [1, 3, 5, 10, 15, 20]
    ndcg_scores = {k: [] for k in Ks}
    recall_scores = {k: [] for k in Ks}
    
    for user_data in data:
        user_id = user_data["user_id"]
        ground_truth = user_data["ground_truth_item_ids"]  # list, thường chỉ 1 item
        predictions = user_data["reranked_item_ids"]       # list đã rerank
        
        if not predictions:
            print(f"Warning: User {user_id} has no predictions → skip")
            continue
        
        for k in Ks:
            ndcg = ndcg_at_k(predictions, ground_truth, k)
            recall = recall_at_k(predictions, ground_truth, k)
            
            ndcg_scores[k].append(ndcg)
            recall_scores[k].append(recall)
    
    # Tính trung bình
    print("="*60)
    print("AVERAGE METRICS ACROSS ALL USERS")
    print("="*60)
    print(f"{'K':>4} | {'NDCG@K':>10} | {'Recall@K':>10}")
    print("-" * 30)
    for k in Ks:
        avg_ndcg = np.mean(ndcg_scores[k])
        avg_recall = np.mean(recall_scores[k])
        print(f"{k:4d} | {avg_ndcg:10.4f} | {avg_recall:10.4f}")
    
    print("\nDone!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze ranking results and compute NDCG@K and Recall@K")
    parser.add_argument(
        "--input", 
        type=str, 
        default="/home/ubuntu/duc.nm195858/AgenticRec_CFmemory/evaluation_results/MIND/zeroshot_users_ranking_no_memory.json",
        help="Path to the JSON file containing all users' ranking results"
    )
    args = parser.parse_args()
    
    analyze_results(args.input)