import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

def parse_args():
    parser = argparse.ArgumentParser(description="Comprehensive Retrieval Analysis with Phenology and Class-level Metrics")
    parser.add_argument("--data_dir", type=str, required=True, 
                        help="Root directory containing all required files (.npy, .csv, .json)")
    parser.add_argument("--output_dir", type=str, default="Retrieval_Comprehensive_Analysis", 
                        help="Directory to save the analysis outputs.")
    parser.add_argument("--max_k_eval", type=int, default=100, 
                        help="Maximum Top-K depth to evaluate for the optimal F1-Score.")
    parser.add_argument("--min_samples", type=int, default=20, 
                        help="Minimum number of samples required to display a class.")
    parser.add_argument("--beta", type=float, default=0.5, 
                        help="Beta value for F-beta score. <1 favors precision (penalizes large N).")
    return parser.parse_args()

def main():
    args = parse_args()


    embeddings_path = os.path.join(args.data_dir, "2_embeddings_list_cleaned.npy")
    annotations_path = os.path.join(args.data_dir, "3_annotations.csv")
    species_mapping_path = os.path.join(args.data_dir, "species_mapping.json")
    flight_mapping_path = os.path.join(args.data_dir, "flight_month_mapping.json")

    required_files = {
        "Embeddings (.npy)": embeddings_path,
        "Annotations (.csv)": annotations_path,
        "Species Dictionary (.json)": species_mapping_path,
        "Flight Dictionary (.json)": flight_mapping_path
    }
    
    print(f"Checking workspace: {args.data_dir}")
    for name, path in required_files.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"{args.data_dir} can not find '{os.path.basename(path)}' ({name})")

    # ==========================================
    # 2. Output Directory Configuration
    # ==========================================
    class_dir = os.path.join(args.output_dir, "Class_PR_and_F1")
    temporal_dir = os.path.join(args.output_dir, "Temporal_Phenology")
    os.makedirs(class_dir, exist_ok=True)
    os.makedirs(temporal_dir, exist_ok=True)

    # ==========================================
    # 3. Data & Dictionary Loading
    # ==========================================
    print("\nLoading datasets and mapping dictionaries...")
    embeddings = np.load(embeddings_path)
    df_anno = pd.read_csv(annotations_path)
    
    with open(species_mapping_path, 'r', encoding='utf-8') as f:
        species_dict = json.load(f)
    with open(flight_mapping_path, 'r', encoding='utf-8') as f:
        flight_to_month = json.load(f)
    
    df_anno['flight'] = df_anno['id'].apply(lambda x: str(x).split('_')[0])
    df_anno['month'] = df_anno['flight'].map(flight_to_month)
    
    if df_anno['month'].isnull().any():
        unmapped = df_anno[df_anno['month'].isnull()]['flight'].unique()
        print(f"Warning: Flights {unmapped} not found in dictionary. They will be ignored in temporal analysis.")
    
    sample_ids = df_anno['id'].values
    labels = df_anno['class_id'].values
    months = df_anno['month'].values
    num_samples = len(embeddings)
    
    print("Computing similarity matrix...")
    sim_matrix = cosine_similarity(embeddings)
    np.fill_diagonal(sim_matrix, -np.inf)
    sorted_indices = np.argsort(-sim_matrix, axis=1)
    
    query_records = []
    common_recalls = np.linspace(0.0, 1.0, 101)

    print("Evaluating individual queries...")
    for i in range(num_samples):
        query_label = labels[i]
        gallery_labels_sorted = labels[sorted_indices[i]][:-1]
        total_targets = np.sum(labels == query_label) - 1
        
        if total_targets <= 0: continue
            
        hits = (gallery_labels_sorted == query_label).astype(int)
        cumulative_hits = np.cumsum(hits)
        
        k_array = np.arange(1, args.max_k_eval + 1)
        precisions_at_k = cumulative_hits[:args.max_k_eval] / k_array
        recalls_at_k = cumulative_hits[:args.max_k_eval] / total_targets
        
        all_k = np.arange(1, num_samples)
        all_precisions = cumulative_hits / all_k
        all_recalls = cumulative_hits / total_targets
        ap = np.sum(all_precisions * hits) / total_targets
        
        interp_p = np.interp(common_recalls, all_recalls, all_precisions)
        interp_p = np.maximum.accumulate(interp_p[::-1])[::-1]
        
        query_records.append({
            'id': sample_ids[i],
            'class_id': query_label,
            'month': months[i],
            'AP': ap,
            'interp_p': interp_p,
            'p_at_k': precisions_at_k,
            'r_at_k': recalls_at_k
        })

    df_results = pd.DataFrame(query_records)
    
    # ==========================================
    # 4. Class-level Analysis (F1 vs F-beta)
    # ==========================================
    print(f"\nGenerating Class-level analysis in {class_dir}...")
    class_summary = []
    beta = args.beta
    beta_sq = beta ** 2
    
    for cls_id, group in df_results.groupby('class_id'):
        if len(group) <= args.min_samples: 
            continue 
            
        species_name = species_dict.get(str(cls_id), f"Unknown Species")
        display_name = f"{species_name} (Class {cls_id})"
            
        class_map = group['AP'].mean()
        mean_interp_p = np.mean(np.vstack(group['interp_p'].values), axis=0)
        mean_p_at_k = np.mean(np.vstack(group['p_at_k'].values), axis=0)
        mean_r_at_k = np.mean(np.vstack(group['r_at_k'].values), axis=0)
        
        denominator_f1 = mean_p_at_k + mean_r_at_k
        f1_scores = np.divide(2 * mean_p_at_k * mean_r_at_k, denominator_f1, 
                              out=np.zeros_like(denominator_f1), where=denominator_f1!=0)
        best_n_f1_idx = np.argmax(f1_scores)
        best_n_f1 = best_n_f1_idx + 1
        max_f1 = f1_scores[best_n_f1_idx]
        
        denominator_fbeta = (beta_sq * mean_p_at_k) + mean_r_at_k
        fbeta_scores = np.divide((1 + beta_sq) * mean_p_at_k * mean_r_at_k, denominator_fbeta, 
                                 out=np.zeros_like(denominator_fbeta), where=denominator_fbeta!=0)
        best_n_fbeta_idx = np.argmax(fbeta_scores)
        best_n_fbeta = best_n_fbeta_idx + 1
        max_fbeta = fbeta_scores[best_n_fbeta_idx]
        
        class_summary.append({
            'class_id': cls_id, 'species_name': species_name, 'mAP': class_map, 
            'best_N_F1': best_n_f1, 'max_F1': max_f1,
            f'best_N_F{beta}': best_n_fbeta, f'max_F{beta}': max_fbeta,
            'sample_count': len(group)
        })
        
        fig, ax1 = plt.subplots(figsize=(9, 6))
        
        ax1.plot(common_recalls, mean_interp_p, 'b-', lw=2, label=f'PR Curve (mAP = {class_map:.3f})')
        ax1.fill_between(common_recalls, mean_interp_p, alpha=0.1, color='blue')
        ax1.set_xlabel('Recall', fontsize=12)
        ax1.set_ylabel('Precision', color='b', fontsize=12)
        ax1.tick_params(axis='y', labelcolor='b')
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
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')
        
        plt.title(f'Performance Profile: {display_name} (n={len(group)})', fontsize=14)
        plt.grid(True, linestyle=':', alpha=0.6)
        
        safe_species_name = species_name.replace(" ", "_").replace("/", "_")
        filename = f"Class_{cls_id}_{safe_species_name}_PR_DualF.png"
        plt.savefig(os.path.join(class_dir, filename), dpi=150, bbox_inches='tight')
        plt.close()

    pd.DataFrame(class_summary).to_csv(os.path.join(class_dir, "class_metrics_summary.csv"), index=False)

    # ==========================================
    # 5. Temporal / Phenology Analysis
    # ==========================================
    print(f"Generating Temporal Phenology analysis in {temporal_dir}...")
    
    df_temporal = df_results.dropna(subset=['month']).copy()
    df_temporal['month'] = df_temporal['month'].astype(int)
    
    valid_classes = df_temporal['class_id'].value_counts()[lambda x: x > args.min_samples].index
    df_top = df_temporal[df_temporal['class_id'].isin(valid_classes)].copy()
    
    df_top['species_label'] = df_top['class_id'].apply(
        lambda x: f"{species_dict.get(str(x), 'Unknown')}"
    )
    
    monthly_map = df_top.groupby('month')['AP'].mean().reset_index()
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=monthly_map, x='month', y='AP', marker='o', color='forestgreen', ax=ax1, linewidth=2.5, markersize=8)
    ax1.set_xlabel('Month (Phenology)', fontsize=14)
    ax1.set_ylabel('Global mAP (Filtered Classes)', fontsize=14, color='forestgreen')
    ax1.set_ylim([0, 1.05])
    
    unique_months = sorted(df_top['month'].unique())
    ax1.set_xticks(unique_months)
    ax1.set_xticklabels([f"Month {int(m)}" for m in unique_months])
    
    plt.title(f'Global Retrieval Performance across Different Months', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(temporal_dir, "Global_Monthly_mAP_Trend.png"), dpi=300, bbox_inches='tight')
    plt.close()

    fig_height = max(6, len(valid_classes) * 0.4) 
    plt.figure(figsize=(12, fig_height))
    
    pivot_table = df_top.pivot_table(values='AP', index='species_label', columns='month', aggfunc='mean')
    
    sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap="YlGnBu", vmin=0, vmax=1.0, 
                cbar_kws={'label': 'Mean Average Precision (mAP)'}, linewidths=.5)
    
    plt.title(f'Species-Specific Retrieval Performance by Month', fontsize=16, pad=15)
    plt.xlabel('Month', fontsize=14)
    plt.ylabel('Tree Species', fontsize=14)
    plt.yticks(rotation=0) 
    
    plt.savefig(os.path.join(temporal_dir, "Species_Month_Phenology_Heatmap.png"), dpi=300, bbox_inches='tight')
    plt.close()

    print("\n" + "="*50)
    print("COMPREHENSIVE ANALYSIS COMPLETED")
    print("="*50)
    print(f"1. Class PR & Dual F-Score Analysis saved to : {class_dir}/")
    print(f"2. Temporal / Phenology trends saved to        : {temporal_dir}/")
    print("="*50)

if __name__ == "__main__":
    main()