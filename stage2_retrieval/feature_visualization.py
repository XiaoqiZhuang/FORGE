import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE

def parse_args():
    parser = argparse.ArgumentParser(description="Sample-level Embedding Visualization (Heatmap & t-SNE)")
    parser.add_argument("--data_dir", type=str, required=True, 
                        help="Root directory containing all required files (.npy, .csv, .json)")
    parser.add_argument("--output_dir", type=str, default="Visualizations", 
                        help="Directory to save the analysis outputs.")
    parser.add_argument("--min_samples", type=int, default=20, 
                        help="Minimum samples required to include a class in visualization.")
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    embeddings_path = os.path.join(args.data_dir, "2_embeddings_list_cleaned.npy")
    annotations_path = os.path.join(args.data_dir, "3_annotations.csv")
    species_mapping_path = os.path.join(args.data_dir, "species_mapping.json")

    print(f"Loading data from {args.data_dir}...")
    embeddings = np.load(embeddings_path)
    df_anno = pd.read_csv(annotations_path)
    
    with open(species_mapping_path, 'r', encoding='utf-8') as f:
        species_dict = json.load(f)

    df_anno['species_name'] = df_anno['class_id'].apply(lambda x: species_dict.get(str(x), f"Class {x}"))

    class_counts = df_anno['class_id'].value_counts()
    valid_classes = class_counts[class_counts >= args.min_samples].index
    
    mask = df_anno['class_id'].isin(valid_classes)
    df_filtered = df_anno[mask].copy()
    emb_filtered = embeddings[mask]
    
    print(f"Filtered to {len(valid_classes)} classes with >= {args.min_samples} samples.")
    print(f"Total samples for visualization: {len(df_filtered)}")

    print("\nGenerating Sorted Block-Diagonal Heatmap...")
    
    sort_order = np.argsort(df_filtered['class_id'].values)
    
    df_sorted = df_filtered.iloc[sort_order].reset_index(drop=True)
    emb_sorted = emb_filtered[sort_order]

    sim_matrix = cosine_similarity(emb_sorted)

    class_sizes = df_sorted['class_id'].value_counts(sort=False).values
    boundaries = np.cumsum(class_sizes)[:-1] 
    unique_species = df_sorted['species_name'].unique()
    
    ticks = np.insert(np.cumsum(class_sizes), 0, 0)
    tick_centers = ticks[:-1] + np.diff(ticks) / 2

    plt.figure(figsize=(12, 10))
    im = plt.imshow(sim_matrix, cmap='viridis', aspect='auto', vmin=0.0, vmax=1.0)
    plt.colorbar(im, label='Cosine Similarity')

    for b in boundaries:
        plt.axhline(y=b, color='red', linestyle='--', linewidth=0.5, alpha=0.7)
        plt.axvline(x=b, color='red', linestyle='--', linewidth=0.5, alpha=0.7)

    plt.xticks(tick_centers, unique_species, rotation=90, fontsize=8)
    plt.yticks(tick_centers, unique_species, fontsize=8)
    
    plt.title(f'Sample-level Similarity Heatmap (Sorted by Species)', fontsize=14, pad=20)
    plt.xlabel('Tree Species', fontsize=12)
    plt.ylabel('Tree Species', fontsize=12)
    
    heatmap_path = os.path.join(args.output_dir, "Sorted_Similarity_Heatmap.png")
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [✓] Saved to {heatmap_path}")

    print("\nGenerating t-SNE Scatter Plot (This might take a minute)...")
    
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    emb_2d = tsne.fit_transform(emb_filtered)

    df_filtered['tsne_x'] = emb_2d[:, 0]
    df_filtered['tsne_y'] = emb_2d[:, 1]

    plt.figure(figsize=(12, 10))
    sns.scatterplot(
        data=df_filtered, 
        x='tsne_x', 
        y='tsne_y', 
        hue='species_name',
        palette='tab20', 
        s=40,            
        alpha=0.8,       
        edgecolor=None
    )
    
    plt.title('t-SNE Visualization of Tree Embeddings', fontsize=16)
    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Tree Species", fontsize=9, title_fontsize=11)
    
    tsne_path = os.path.join(args.output_dir, "tSNE_Embeddings.png")
    plt.savefig(tsne_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [✓] Saved to {tsne_path}")

    print("\n" + "="*50)
    print("VISUALIZATIONS COMPLETED SUCCESSFULLY")
    print("="*50)

if __name__ == "__main__":
    main()