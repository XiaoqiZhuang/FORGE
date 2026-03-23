import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

# Mapping dictionary from flight number to month
FLIGHT_TO_MONTH = {
    '01': 3, '02': 4, '03': 5, '04': 4, '05': 4, 
    '06': 6, '07': 5, '08': 5, '09': 6, '10': 6, 
    '11': 7, '12': 8, '13': 8, '14': 8, '15': 8, 
    '16': 9, '17': 10, '18': 10, '19': 10, 
    '20': 11, '21': 11, '22': 11
}

def parse_args():
    parser = argparse.ArgumentParser(description="Comprehensive Retrieval Analysis with Phenology and Class-level Metrics")
    parser.add_argument("--embeddings_path", type=str, required=True, 
                        help="Path to the embeddings .npy file.")
    parser.add_argument("--annotations_path", type=str, required=True, 
                        help="Path to the annotations .csv file.")
    parser.add_argument("--output_dir", type=str, default="Retrieval_Comprehensive_Analysis", 
                        help="Directory to save the analysis outputs.")
    parser.add_argument("--max_k_eval", type=int, default=100, 
                        help="Maximum Top-K depth to evaluate for the optimal F1-Score.")
    return parser.parse_args()

def main():
    args = parse_args()

    # ==========================================
    # 1. Output Directory Configuration
    # ==========================================
    class_dir = os.path.join(args.output_dir, "Class_PR_and_F1")
    temporal_dir = os.path.join(args.output_dir, "Temporal_Phenology")
    os.makedirs(class_dir, exist_ok=True)
    os.makedirs(temporal_dir, exist_ok=True)

    # ==========================================
    # 2. Data Loading & Preprocessing
    # ==========================================
    print("Loading data...")
    embeddings = np.load(args.embeddings_path)
    df_anno = pd.read_csv(args.annotations_path)
    
    # Extract flight number and map to month
    df_anno['flight'] = df_anno['id'].apply(lambda x: str(x).split('_')[0])
    df_anno['month'] = df_anno['flight'].map(FLIGHT_TO_MONTH)
    
    if df_anno['month'].isnull().any():
        print("Warning: Some IDs have unknown flight numbers. They will be ignored in temporal analysis.")
    
    sample_ids = df_anno['id'].values
    labels = df_anno['class_id'].values
    months = df_anno['month'].values
    num_samples = len(embeddings)
    
    print("Computing similarity matrix...")
    sim_matrix = cosine_similarity(embeddings)
    # Set diagonal to -inf to exclude self-matching
    np.fill_diagonal(sim_matrix, -np.inf)
    sorted_indices = np.argsort(-sim_matrix, axis=1)
    
    query_records = []
    
    # Standardized recall points for interpolating PR curves
    common_recalls = np.linspace(0.0, 1.0, 101)

    print("Evaluating individual queries...")
    for i in range(num_samples):
        query_label = labels[i]
        gallery_labels_sorted = labels[sorted_indices[i]][:-1]
        total_targets = np.sum(labels == query_label) - 1
        
        # Skip if there are no other instances of the same class
        if total_targets <= 0: continue
            
        hits = (gallery_labels_sorted == query_label).astype(int)
        cumulative_hits = np.cumsum(hits)
        
        # Calculate P and R from Top-1 to max_k_eval
        k_array = np.arange(1, args.max_k_eval + 1)
        precisions_at_k = cumulative_hits[:args.max_k_eval] / k_array
        recalls_at_k = cumulative_hits[:args.max_k_eval] / total_targets
        
        # Calculate global P and R for PR curve interpolation
        all_k = np.arange(1, num_samples)
        all_precisions = cumulative_hits / all_k
        all_recalls = cumulative_hits / total_targets
        ap = np.sum(all_precisions * hits) / total_targets
        
        # Interpolate precision and ensure it is monotonically decreasing
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
    # 3. Class-level Analysis: PR Curves and Optimal N
    # ==========================================
    print(f"Generating Class-level analysis in {class_dir}...")
    class_summary = []
    
    for cls_id, group in df_results.groupby('class_id'):
        # Ignore classes with very few samples for statistical significance
        if len(group) < 5: continue 
            
        class_map = group['AP'].mean()
        
        # Calculate the mean PR, P@K, and R@K for the class
        mean_interp_p = np.mean(np.vstack(group['interp_p'].values), axis=0)
        mean_p_at_k = np.mean(np.vstack(group['p_at_k'].values), axis=0)
        mean_r_at_k = np.mean(np.vstack(group['r_at_k'].values), axis=0)
        
        # Calculate F1-Score to find the optimal N
        denominator = mean_p_at_k + mean_r_at_k
        # Avoid division by zero
        f1_scores = np.divide(2 * mean_p_at_k * mean_r_at_k, denominator, 
                              out=np.zeros_like(denominator), where=denominator!=0)
        
        best_n_idx = np.argmax(f1_scores)
        best_n = best_n_idx + 1
        max_f1 = f1_scores[best_n_idx]
        
        class_summary.append({
            'class_id': cls_id, 'mAP': class_map, 
            'best_N': best_n, 'max_F1': max_f1, 'sample_count': len(group)
        })
        
        # --- Plot dual-axis chart: PR curve and F1 curve ---
        fig, ax1 = plt.subplots(figsize=(8, 6))
        
        # Left axis: PR Curve
        ax1.plot(common_recalls, mean_interp_p, 'b-', lw=2, label=f'PR Curve (AUC/mAP = {class_map:.3f})')
        ax1.fill_between(common_recalls, mean_interp_p, alpha=0.1, color='blue')
        ax1.set_xlabel('Recall', fontsize=12)
        ax1.set_ylabel('Precision', color='b', fontsize=12)
        ax1.tick_params(axis='y', labelcolor='b')
        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        
        # Right axis: F1 Curve
        ax2 = ax1.twinx()
        ax2.plot(mean_r_at_k, f1_scores, 'r--', lw=2, label='F1-Score vs Recall')
        ax2.plot(mean_r_at_k[best_n_idx], max_f1, 'r*', markersize=12, label=f'Best N={best_n} (F1={max_f1:.2f})')
        ax2.set_ylabel('F1-Score', color='r', fontsize=12)
        ax2.tick_params(axis='y', labelcolor='r')
        ax2.set_ylim([0.0, 1.05])
        
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')
        
        plt.title(f'Performance Profile: Class {cls_id} (n={len(group)})')
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.savefig(os.path.join(class_dir, f"Class_{cls_id}_PR_F1.png"), dpi=150, bbox_inches='tight')
        plt.close()

    pd.DataFrame(class_summary).to_csv(os.path.join(class_dir, "class_metrics_summary.csv"), index=False)

    # ==========================================
    # 4. Temporal / Phenology Analysis
    # ==========================================
    print(f"Generating Temporal Phenology analysis in {temporal_dir}...")
    
    # Filter out rows where month could not be mapped (NaN)
    df_temporal = df_results.dropna(subset=['month']).copy()
    df_temporal['month'] = df_temporal['month'].astype(int)
    
    # --- 4.1 Global monthly trend line chart ---
    monthly_map = df_temporal.groupby('month')['AP'].mean().reset_index()
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=monthly_map, x='month', y='AP', marker='o', color='forestgreen', ax=ax1, linewidth=2.5, markersize=8)
    ax1.set_xlabel('Month (Phenology)', fontsize=14)
    ax1.set_ylabel('Global mAP', fontsize=14, color='forestgreen')
    ax1.set_ylim([0, 1.05])
    
    unique_months = sorted(df_temporal['month'].unique())
    ax1.set_xticks(unique_months)
    ax1.set_xticklabels([f"Month {m}" for m in unique_months])
    
    plt.title('Global Retrieval Performance across Different Months', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(temporal_dir, "Global_Monthly_mAP_Trend.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # --- 4.2 Species x Month phenology heatmap ---
    # Select top 15 dominant tree species by sample size to keep heatmap readable
    top_classes = df_temporal['class_id'].value_counts().nlargest(15).index
    df_top = df_temporal[df_temporal['class_id'].isin(top_classes)]
    
    pivot_table = df_top.pivot_table(values='AP', index='class_id', columns='month', aggfunc='mean')
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap="YlGnBu", vmin=0, vmax=1.0, 
                cbar_kws={'label': 'Mean Average Precision (mAP)'}, linewidths=.5)
    plt.title('Species-Specific Retrieval Performance by Month (Phenology)', fontsize=16)
    plt.xlabel('Month', fontsize=14)
    plt.ylabel('Tree Species (Class ID)', fontsize=14)
    plt.savefig(os.path.join(temporal_dir, "Species_Month_Phenology_Heatmap.png"), dpi=300, bbox_inches='tight')
    plt.close()

    print("\n" + "="*50)
    print("COMPREHENSIVE ANALYSIS COMPLETED")
    print("="*50)
    print(f"1. Class PR curves & F1 Optimization saved to : {class_dir}/")
    print(f"2. Temporal / Phenology trends saved to       : {temporal_dir}/")
    print("="*50)

if __name__ == "__main__":
    main()