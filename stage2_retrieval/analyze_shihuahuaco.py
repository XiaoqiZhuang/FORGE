import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

data_dir = "D:/Forge/Tree_Features"
target_species = "Shihuahuaco"  

embeddings = np.load(os.path.join(data_dir, "2_embeddings_list_cleaned.npy"))
df_anno = pd.read_csv(os.path.join(data_dir, "3_annotations.csv"))
with open(os.path.join(data_dir, "species_mapping.json"), 'r', encoding='utf-8') as f:
    species_dict = json.load(f)
with open(os.path.join(data_dir, "flight_month_mapping.json"), 'r', encoding='utf-8') as f:
    flight_to_month = json.load(f)

df_anno['species_name'] = df_anno['class_id'].apply(lambda x: species_dict.get(str(x), f"Class {x}"))
df_anno['flight'] = df_anno['id'].apply(lambda x: str(x).split('_')[0])
df_anno['month'] = df_anno['flight'].map(flight_to_month)

mask = (df_anno['species_name'] == target_species) & (df_anno['month'].notnull())
df_target = df_anno[mask].copy()
emb_target = embeddings[mask]

print(f"Found {len(df_target)} samples for {target_species}.")

perp = min(30, len(df_target) - 1)
tsne = TSNE(n_components=2, perplexity=perp, random_state=42, max_iter=1000)
emb_2d = tsne.fit_transform(emb_target)

df_target['tsne_x'] = emb_2d[:, 0]
df_target['tsne_y'] = emb_2d[:, 1]
df_target['month_str'] = df_target['month'].astype(int).astype(str) + " Month"

plt.figure(figsize=(10, 8))
sns.scatterplot(
    data=df_target.sort_values('month'),
    x='tsne_x', 
    y='tsne_y', 
    hue='month_str',
    palette='Set1', 
    s=80, 
    alpha=0.8,
    edgecolor='k'
)

plt.title(f'Intra-class Variance: t-SNE of {target_species} by Month', fontsize=16)
plt.xlabel('t-SNE Dim 1', fontsize=12)
plt.ylabel('t-SNE Dim 2', fontsize=12)
plt.legend(title="Phenology (Month)", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.5)

output_file = f"tSNE_{target_species}_by_Month.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"Saved highly detailed visualization to {output_file}")