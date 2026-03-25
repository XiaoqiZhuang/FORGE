import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

def parse_args():
    parser = argparse.ArgumentParser(description="Real Inference Scenario Evaluation (Intra-flight Retrieval)")
    parser.add_argument("--data_dir", type=str, required=True, 
                        help="Root directory containing all required files (.npy, .csv, .json)")
    parser.add_argument("--output_dir", type=str, default="Real_Inference_Evaluation", 
                        help="Directory to save the analysis outputs.")
    parser.add_argument("--target_species", type=str, default="Shihuahuaco", 
                        help="Target species name to evaluate.")
    parser.add_argument("--max_k_eval", type=int, default=100, 
                        help="Maximum Top-K depth to evaluate.")
    parser.add_argument("--beta", type=float, default=0.5, 
                        help="Beta value for F-beta score.")
    return parser.parse_args()

def main():
    args = parse_args()
    
    output_dir = os.path.join(args.output_dir, f"{args.target_species}_per_Flight")
    os.makedirs(output_dir, exist_ok=True)

    embeddings_path = os.path.join(args.data_dir, "2_embeddings_list_cleaned.npy")
    annotations_path = os.path.join(args.data_dir, "3_annotations.csv")
    species_mapping_path = os.path.join(args.data_dir, "species_mapping.json")
    flight_mapping_path = os.path.join(args.data_dir, "flight_month_mapping.json")

    print(f"Loading data from {args.data_dir}...")
    embeddings = np.load(embeddings_path)
    df_anno = pd.read_csv(annotations_path)
    
    with open(species_mapping_path, 'r', encoding='utf-8') as f:
        species_dict = json.load(f)
        
    with open(flight_mapping_path, 'r', encoding='utf-8') as f:
        flight_to_month = json.load(f)

    target_class_id = None
    for cid, name in species_dict.items():
        if name.strip().lower() == args.target_species.strip().lower():
            target_class_id = int(cid)
            break
            
    if target_class_id is None:
        raise ValueError(f" '{args.target_species}' not found in species mapping. Please check the name and try again.")
    
    print(f"Target Species: {args.target_species} (Class ID: {target_class_id})")

    df_anno['flight'] = df_anno['id'].apply(lambda x: str(x).split('_')[0])
    
    summary_records = []
    common_recalls = np.linspace(0.0, 1.0, 101)

    grouped = df_anno.groupby('flight')
    print(f"\nFound {len(grouped)} flights. Starting isolated evaluations...")

    for flight_id, flight_df in grouped:
        flight_indices = flight_df.index.values
        flight_labels = flight_df['class_id'].values
        
        target_count = np.sum(flight_labels == target_class_id)
        total_trees = len(flight_labels)
        
        flight_date = flight_to_month.get(str(flight_id), "Unknown Date")
        
        if target_count < 2:
            print(f"  - Skipping Flight {flight_id} (Date: {flight_date}): Not enough {args.target_species} trees (only {target_count} trees).")
            continue
            
        print(f"  + Evaluating Flight {flight_id} (Date: {flight_date}): Total trees {total_trees}, including {args.target_species} {target_count}.")
        
        flight_emb = embeddings[flight_indices]
        sim_matrix = cosine_similarity(flight_emb)
        np.fill_diagonal(sim_matrix, -np.inf)
        sorted_indices = np.argsort(-sim_matrix, axis=1) 
        
        query_indices = np.where(flight_labels == target_class_id)[0]
        flight_query_records = []
        max_k = min(args.max_k_eval, total_trees - 1)
        k_array = np.arange(1, max_k + 1)
        all_k = np.arange(1, total_trees)

        for q_idx in query_indices:
            gallery_labels_sorted = flight_labels[sorted_indices[q_idx]][:-1]
            total_targets = target_count - 1
            
            hits = (gallery_labels_sorted == target_class_id).astype(int)
            cumulative_hits = np.cumsum(hits)
            
            precisions_at_k = cumulative_hits[:max_k] / k_array
            recalls_at_k = cumulative_hits[:max_k] / total_targets
            
            all_precisions = cumulative_hits / all_k
            all_recalls = cumulative_hits / total_targets
            ap = np.sum(all_precisions * hits) / total_targets
            
            interp_p = np.interp(common_recalls, all_recalls, all_precisions)
            interp_p = np.maximum.accumulate(interp_p[::-1])[::-1]
            
            flight_query_records.append({
                'AP': ap,
                'interp_p': interp_p,
                'p_at_k': precisions_at_k,
                'r_at_k': recalls_at_k
            })
            
        mean_ap = np.mean([r['AP'] for r in flight_query_records])
        mean_interp_p = np.mean(np.vstack([r['interp_p'] for r in flight_query_records]), axis=0)
        mean_p_at_k = np.mean(np.vstack([r['p_at_k'] for r in flight_query_records]), axis=0)
        mean_r_at_k = np.mean(np.vstack([r['r_at_k'] for r in flight_query_records]), axis=0)
        
        p_at_5 = mean_p_at_k[4] if len(mean_p_at_k) >= 5 else np.nan
        p_at_10 = mean_p_at_k[9] if len(mean_p_at_k) >= 10 else np.nan
        
        beta = args.beta
        beta_sq = beta ** 2
        
        denom_f1 = mean_p_at_k + mean_r_at_k
        f1_scores = np.divide(2 * mean_p_at_k * mean_r_at_k, denom_f1, out=np.zeros_like(denom_f1), where=denom_f1!=0)
        best_n_f1_idx = np.argmax(f1_scores)
        best_n_f1 = best_n_f1_idx + 1
        max_f1 = f1_scores[best_n_f1_idx]
        
        denom_fbeta = (beta_sq * mean_p_at_k) + mean_r_at_k
        fbeta_scores = np.divide((1 + beta_sq) * mean_p_at_k * mean_r_at_k, denom_fbeta, out=np.zeros_like(denom_fbeta), where=denom_fbeta!=0)
        best_n_fbeta_idx = np.argmax(fbeta_scores)
        best_n_fbeta = best_n_fbeta_idx + 1
        max_fbeta = fbeta_scores[best_n_fbeta_idx]
        
        summary_records.append({
            'flight_id': flight_id,
            'flight_date': flight_date,  
            'species': args.target_species,
            'total_trees_in_flight': total_trees,
            'target_trees_in_flight': target_count,
            'mAP': mean_ap,
            'P@5': p_at_5,
            'P@10': p_at_10,
            'best_N_F1': best_n_f1,
            'max_F1': max_f1,
            f'best_N_F{beta}': best_n_fbeta,
            f'max_F{beta}': max_fbeta
        })
        
        fig, ax1 = plt.subplots(figsize=(9, 6))
        
        ax1.plot(common_recalls, mean_interp_p, 'b-', lw=2, label=f'PR Curve (mAP = {mean_ap:.3f})')
        ax1.plot([], [], ' ', label=f'  ├ P@5  = {p_at_5:.3f}')
        ax1.plot([], [], ' ', label=f'  └ P@10 = {p_at_10:.3f}')
        ax1.fill_between(common_recalls, mean_interp_p, alpha=0.1, color='blue')
        
        ax1.set_xlabel('Recall', fontsize=12)
        ax1.set_ylabel('Precision', color='b', fontsize=12)
        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        
        ax2 = ax1.twinx()
        ax2.plot(mean_r_at_k, f1_scores, 'r--', lw=2, alpha=0.7, label='F1-Score')
        ax2.plot(mean_r_at_k[best_n_f1_idx], max_f1, 'r*', markersize=12, label=f'Best F1 (N={best_n_f1})')
        ax2.plot(mean_r_at_k, fbeta_scores, 'g-.', lw=2, alpha=0.9, label=f'F{beta}-Score')
        ax2.plot(mean_r_at_k[best_n_fbeta_idx], max_fbeta, 'g^', markersize=10, label=f'Best F{beta} (N={best_n_fbeta})')
        ax2.set_ylabel('F-Score Metrics', color='k', fontsize=12)
        ax2.set_ylim([0.0, 1.05])
        
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right', fontsize=10)
        
        plt.title(f'Real Inference: {args.target_species} (Flight {flight_id} | Date: Month {flight_date})\nTarget Trees: {target_count} / Total Pool: {total_trees}', fontsize=13)
        plt.grid(True, linestyle=':', alpha=0.6)
        
        plt.savefig(os.path.join(output_dir, f"Flight_{flight_id}_PR_Curve.png"), dpi=150, bbox_inches='tight')
        plt.close()

    if summary_records:
        df_summary = pd.DataFrame(summary_records)
        df_summary.to_csv(os.path.join(output_dir, f"Summary_Metrics_{args.target_species}.csv"), index=False)

    else:
        print(f"\nNo flights had enough {args.target_species} trees for evaluation. No summary file generated.")
if __name__ == "__main__":
    main()