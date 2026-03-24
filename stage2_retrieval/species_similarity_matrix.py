import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity


DATASET_DIR = "D:/Forge/Tree_Features"
EMBEDDINGS_PATH = os.path.join(DATASET_DIR, "2_embeddings_list_cleaned.npy")
ANNOTATIONS_PATH = os.path.join(DATASET_DIR, "3_annotations.csv")

SPECIES_DICT_PATH = os.path.join(DATASET_DIR, "species_mapping.json" )

OUTPUT_DIR = "Matrix_Analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MIN_SAMPLES = 20

def main():
    print("Loading data...")
    embeddings = np.load(EMBEDDINGS_PATH)
    df_anno = pd.read_csv(ANNOTATIONS_PATH)
    
    if len(embeddings) != len(df_anno):
        raise ValueError("Mismatch between embeddings and annotations count.")

    with open(SPECIES_DICT_PATH, 'r', encoding='utf-8') as f:
        species_mapping = json.load(f)

    class_counts = df_anno['class_id'].value_counts()
    valid_classes = class_counts[class_counts >= MIN_SAMPLES].index.sort_values()
    
    prototypes = []
    class_labels = []

    for class_id in valid_classes:
        indices = df_anno.index[df_anno['class_id'] == class_id].tolist()
        class_embeddings = embeddings[indices]
        class_prototype = np.mean(class_embeddings, axis=0)
        
        prototypes.append(class_prototype)
        
        species_name = species_mapping.get(str(class_id), f"Class {class_id}")

        class_labels.append(f"{species_name}")
        
    prototypes = np.array(prototypes)

    sim_matrix = cosine_similarity(prototypes)
    np.save(os.path.join(OUTPUT_DIR, "species_prototype_similarity.npy"), sim_matrix)
    
    fig_size = max(10, len(valid_classes) * 0.35)
    plt.figure(figsize=(fig_size, fig_size))
    
    img = plt.imshow(sim_matrix, cmap='viridis', aspect='auto')
    plt.colorbar(img, label='Cosine Similarity')
    
    ticks = np.arange(len(class_labels))
    plt.xticks(ticks, class_labels, rotation=90, fontsize=9)
    plt.yticks(ticks, class_labels, fontsize=9)

    plt.tight_layout()
    
    plot_path = os.path.join(OUTPUT_DIR, "Species_Similarity_Matrix_Heatmap.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
if __name__ == "__main__":
    main()