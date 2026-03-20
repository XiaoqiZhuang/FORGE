import os
import time
import argparse
import torch
import numpy as np
import rasterio
import geopandas as gpd
from PIL import Image
from rasterio.mask import mask
from shapely.geometry import mapping
from sklearn.metrics.pairwise import cosine_similarity
import torchvision.transforms as T

def parse_args():
    parser = argparse.ArgumentParser(description="Automated Tree Crown Retrieval using DINOv2.")
    
    # Input/Output arguments
    parser.add_argument("--tif_path", type=str, required=True,
                        help="Path to the original RGB TIFF orthomosaic.")
    parser.add_argument("--candidate_pool", type=str, required=True,
                        help="Path to the YOLO candidate pool GPKG/SHP.")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save the retrieval results (default: same as TIF path).")
    
    # Retrieval arguments
    parser.add_argument("--query_fid", type=int, required=True,
                        help="The FID (Feature ID) of the query tree in QGIS/GPKG.")
    
    # Model arguments
    parser.add_argument("--model_size", type=str, default="dinov2_vits14",
                        choices=["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14"],
                        help="DINOv2 model size architecture.")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Compute device (e.g., 'cuda:0' or 'cpu').")
    
    return parser.parse_args()

def get_dino_model(size, device):
    """Initialize and load the DINOv2 model."""
    print(f"Loading {size} model to {device}...")
    model = torch.hub.load('facebookresearch/dinov2', size).to(device)
    model.eval()
    return model

def get_crop_transforms():
    """Define standard preprocessing pipeline for DINOv2."""
    return T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

def extract_embedding_with_gray_mask(geometry, src_raster, model, transforms, device):
    """Extract DINOv2 feature vector using a masked polygon crop."""
    try:
        out_image, _ = mask(src_raster, [mapping(geometry)], crop=True, filled=False)
        
        if out_image.shape[0] < 3:
             return None
        rgb_image = out_image[:3, :, :] 
        
        gray_fill_value = 124 
        gray_bg = np.full_like(rgb_image.data, gray_fill_value)

        final_img_array = np.where(rgb_image.mask, gray_bg, rgb_image.data)
        
        img_array_hwc = np.transpose(final_img_array, (1, 2, 0))
        pil_img = Image.fromarray(img_array_hwc.astype('uint8'))
        
        img_tensor = transforms(pil_img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            emb = model(img_tensor)
            return torch.nn.functional.normalize(emb, dim=1).cpu().numpy()
            
    except Exception as e:
        print(f"Warning: Failed to extract embedding for geometry: {e}")
        return None

def main():
    args = parse_args()
    
    if not os.path.exists(args.tif_path) or not os.path.exists(args.candidate_pool):
        print("Error: Invalid TIF or Candidate Pool path provided.")
        return

    print("Initializing retrieval pipeline...")
    start_time = time.time()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = get_dino_model(args.model_size, device)
    transforms = get_crop_transforms()
    
    print(f"Loading candidate pool from: {args.candidate_pool}")
    gdf = gpd.read_file(args.candidate_pool)
    
    if gdf.empty:
        print("Error: Candidate pool is empty.")
        return

    fid_col = next((col for col in gdf.columns if col.lower() == 'fid'), None)
    
    if fid_col is not None:
        query_rows = gdf[gdf[fid_col] == args.query_fid]
    else:
        query_rows = gdf[gdf.index == args.query_fid]

    if query_rows.empty:
        print(f"Error: could not find tree with FID {args.query_fid}! Please check the attribute table in QGIS.")
        return
        
    query_geometry = query_rows.iloc[0].geometry
    actual_row_index = query_rows.index[0]
    
    print(f"Defining query tree with FID: {args.query_fid} (Internal Row Index: {actual_row_index})")
    
    target_pool_gdf = gdf.drop(index=actual_row_index).reset_index(drop=True)    
    embeddings = []
    
    with rasterio.open(args.tif_path) as src:
        if target_pool_gdf.crs and target_pool_gdf.crs != src.crs:
            print("Aligning GDF CRS with Raster CRS...")
            target_pool_gdf = target_pool_gdf.to_crs(src.crs)
            
        query_emb = extract_embedding_with_gray_mask(query_geometry, src, model, transforms, device)
        if query_emb is None:
            print("Error: Failed to extract query embedding.")
            return
            
        print(f"Extracting features for {len(target_pool_gdf)} candidate trees...")
        for idx, row in target_pool_gdf.iterrows():
            target_geom = row.geometry
            target_emb = extract_embedding_with_gray_mask(target_geom, src, model, transforms, device)
            
            if target_emb is not None:
                embeddings.append(target_emb.squeeze())
            else:
                embeddings.append(np.zeros(model.embed_dim, dtype=np.float32))

    embeddings_matrix = np.array(embeddings)
    query_emb_2d = query_emb.reshape(1, -1)
    
    print("Calculating similarity scores...")
    sim_scores = cosine_similarity(query_emb_2d, embeddings_matrix)
    target_pool_gdf['Similarity'] = sim_scores.squeeze()
    
    results_gdf = target_pool_gdf.sort_values(by='Similarity', ascending=False).reset_index(drop=True)
    
    output_dir = args.output_dir if args.output_dir else os.path.dirname(args.tif_path)
    base_name = os.path.basename(args.tif_path).rsplit('.', 1)[0]
    out_path = os.path.join(output_dir, f"{base_name}_RETRIEVAL_Results_{args.query_fid}_{int(time.time())}.gpkg")
    
    print(f"Saving sorted results to: {out_path}")
    results_gdf.to_file(out_path, driver="GPKG")
    
    elapsed = time.time() - start_time
    print(f"Time Elapsed      : {elapsed:.2f} seconds")
    print("="*50)

if __name__ == "__main__":
    main()