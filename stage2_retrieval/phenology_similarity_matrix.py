import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

DATASET_DIR = "D:/Forge/Tree_Features"
EMBEDDINGS_PATH = os.path.join(DATASET_DIR, "2_embeddings_list_cleaned.npy")
ANNOTATIONS_PATH = os.path.join(DATASET_DIR, "3_annotations.csv")
SPECIES_DICT_PATH = os.path.join(DATASET_DIR, "species_mapping.json")
FLIGHT_TO_MONTH = os.path.join(DATASET_DIR, "flight_month_mapping.json")

with open(FLIGHT_TO_MONTH, 'r', encoding='utf-8') as f:
    flight_to_month = json.load(f)


OUTPUT_DIR = "Matrix_Analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_CLASS_ID = 10  #  10: Shihuahuaco

MIN_SAMPLES_PER_MONTH = 3 

ID_COLUMN_NAME = 'id' 

def main():
    print(f"Loading data to analyze phenology for Class {TARGET_CLASS_ID}...")
    embeddings = np.load(EMBEDDINGS_PATH)
    df_anno = pd.read_csv(ANNOTATIONS_PATH)
    
    species_name = f"Class {TARGET_CLASS_ID}"
    if os.path.exists(SPECIES_DICT_PATH):
        with open(SPECIES_DICT_PATH, 'r', encoding='utf-8') as f:
            species_mapping = json.load(f)
            species_name = species_mapping.get(str(TARGET_CLASS_ID), species_name)


    df_anno['flight_prefix'] = df_anno[ID_COLUMN_NAME].astype(str).str[:2]
    df_anno['month'] = df_anno['flight_prefix'].map(flight_to_month)
    

    df_target = df_anno[df_anno['class_id'] == TARGET_CLASS_ID]
    print(f"Found {len(df_target)} total samples for {species_name}.")


    available_months = sorted(df_target['month'].unique().astype(int))
    
    valid_months = []
    prototypes = []

    print("Computing mean embedding (prototype) for each month...")
    for month in available_months:
        month_indices = df_target[df_target['month'] == month].index.tolist()
        
        sample_count = len(month_indices)
        if sample_count < MIN_SAMPLES_PER_MONTH:
            print(f"  ({sample_count} < {MIN_SAMPLES_PER_MONTH})")
            continue
            
        month_embeddings = embeddings[month_indices]
        month_prototype = np.mean(month_embeddings, axis=0)
        
        prototypes.append(month_prototype)
        valid_months.append(f"Month {month}")

    prototypes = np.array(prototypes)

    print("Computing Month-to-Month similarity matrix...")
    sim_matrix = cosine_similarity(prototypes)

    print("Generating heatmap visualization...")
    plt.figure(figsize=(8, 6))
    
    img = plt.imshow(sim_matrix, cmap='magma', aspect='auto', vmin=np.min(sim_matrix), vmax=1.0)
    plt.colorbar(img, label='Cosine Similarity')
    
    ticks = np.arange(len(valid_months))
    plt.xticks(ticks, valid_months, fontsize=11)
    plt.yticks(ticks, valid_months, fontsize=11)
    
    for i in range(len(valid_months)):
        for j in range(len(valid_months)):
            plt.text(j, i, f"{sim_matrix[i, j]:.2f}", 
                     ha="center", va="center", 
                     color="white" if sim_matrix[i, j] < (np.max(sim_matrix) - 0.1) else "black")

    plt.title(f'Intra-Class Phenological Similarity\n{species_name} (Class {TARGET_CLASS_ID})', fontsize=14, pad=15)
    plt.xlabel('Timeline', fontsize=12)
    plt.ylabel('Timeline', fontsize=12)
    
    plt.tight_layout()
    
    plot_path = os.path.join(OUTPUT_DIR, f"Phenology_Similarity_{TARGET_CLASS_ID}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    

if __name__ == "__main__":
    main()