import os
import glob
import cv2
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.windows import Window
import random
from shapely.geometry import box
from sklearn.model_selection import train_test_split
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Generate YOLO instance segmentation dataset from orthomosaics and shapefiles.")
    
    # Input/Output paths
    parser.add_argument("--tif_dir", type=str, default="data/", help="Directory containing raw TIF orthomosaics.")
    parser.add_argument("--shp_path", type=str, default="./copas-2023/copas_2023_condatos_vs2.shp", help="Path to the polygon shapefile.")
    parser.add_argument("--output_dir", type=str, default="data/yolo_dataset/", help="Output directory for the YOLO dataset.")
    
    # Image processing parameters
    parser.add_argument("--slice_size", type=int, default=1024, help="Size of the sliding window slice (e.g., 1024x1024).")
    parser.add_argument("--min_visible_ratio", type=float, default=0.30, help="Minimum visible area ratio to keep a truncated polygon.")
    parser.add_argument("--black_tolerance", type=float, default=0.05, help="Tolerance for black/white padding pixels (0.0 to 1.0).")
    
    # 🌟 Target and Data Splitting
    parser.add_argument("--species_column", type=str, default="NOMBRE_COM", help="The column name in the shapefile containing the species name.")
    parser.add_argument("--target_species", type=str, default="Shihuahuaco", help="The target species name in the shapefile.")
    parser.add_argument("--val_split", type=float, default=0.2, help="Validation set ratio.")
    
    # Augmentation parameters
    parser.add_argument("--train_target_aug", type=int, default=8, help="Jittering augmentation factor for target species in training set.")
    parser.add_argument("--train_bg_aug", type=int, default=1, help="Jittering augmentation factor for background trees in training set.")
    
    return parser.parse_args()

def shapely_to_yolo_polygon(clipped_geom, window_transform, slice_size):
    """
    Convert a clipped Shapely polygon to YOLO normalized coordinates.
    """
    if clipped_geom.is_empty: 
        return None
        
    # Handle MultiPolygons caused by complex edge clipping
    if clipped_geom.geom_type == 'MultiPolygon':
        clipped_geom = max(clipped_geom.geoms, key=lambda a: a.area)
        
    if clipped_geom.geom_type != 'Polygon': 
        return None

    norm_coords = []
    for x, y in clipped_geom.exterior.coords:
        # Transform geographic coordinates to pixel coordinates
        col, row = ~window_transform * (x, y)
        
        # Normalize to [0, 1] and clip to ensure they stay within boundaries
        norm_x = max(0.0, min(1.0, col / slice_size))
        norm_y = max(0.0, min(1.0, row / slice_size))
        norm_coords.extend([f"{norm_x:.5f}", f"{norm_y:.5f}"])
        
    return " ".join(norm_coords)

def generate_real_dataset(target_gdf, context_gdf, tif_files, split_name, args, augments_per_tree=1):
    """
    Core engine: Generates image slices with spatial jittering and extracts YOLO labels.
    """
    slice_counter = 0
    slice_size = args.slice_size
    
    for tif_path in tif_files:
        with rasterio.open(tif_path) as src:
            # Spatial filtering: extract polygons within the current TIF boundaries
            tif_bounds = src.bounds
            target_in_tif = target_gdf.to_crs(src.crs).cx[tif_bounds.left:tif_bounds.right, tif_bounds.bottom:tif_bounds.top]
            context_in_tif = context_gdf.to_crs(src.crs).cx[tif_bounds.left:tif_bounds.right, tif_bounds.bottom:tif_bounds.top]
            
            if target_in_tif.empty: 
                continue
            
            for idx, row in target_in_tif.iterrows():
                centroid = row.geometry.centroid
                center_row, center_col = src.index(centroid.x, centroid.y)
                
                for aug_idx in range(augments_per_tree):
                    # Spatial jittering: float the target tree between 20% and 80% of the slice
                    offset_x = random.randint(int(slice_size * 0.2), int(slice_size * 0.8))
                    offset_y = random.randint(int(slice_size * 0.2), int(slice_size * 0.8))
                    
                    slice_min_col = int(center_col - offset_x)
                    slice_min_row = int(center_row - offset_y)
                    
                    # Boundary check
                    if (slice_min_col < 0 or slice_min_row < 0 or 
                        slice_min_col + slice_size > src.width or 
                        slice_min_row + slice_size > src.height):
                        continue
                        
                    # Read image slice
                    window = Window(slice_min_col, slice_min_row, slice_size, slice_size)
                    bg_slice = src.read([1, 2, 3], window=window).transpose(1, 2, 0).copy()
                    bg_slice = cv2.cvtColor(bg_slice, cv2.COLOR_RGB2BGR)
                    
                    # Filter invalid background (detect pure black/white padding areas)
                    black_pixels = np.all(bg_slice == [0, 0, 0], axis=-1)
                    white_pixels = np.all(bg_slice > [250, 250, 250], axis=-1)
                    invalid_ratio = np.sum(black_pixels | white_pixels) / (slice_size * slice_size)

                    if invalid_ratio > args.black_tolerance:
                        continue
                    
                    slice_bounds_geom = box(*src.window_bounds(window))
                    window_transform = src.window_transform(window)
                    yolo_labels = []
                    
                    # Find all co-occurring trees within this 1024x1024 bounding box
                    trees_in_slice = context_in_tif[context_in_tif.intersects(slice_bounds_geom)]
                    
                    for _, t_row in trees_in_slice.iterrows():
                        t_geom = t_row.geometry
                        t_class = 0 if t_row[args.species_column] == args.target_species else 1
                        
                        # Edge truncation and visibility check
                        clipped_geom = t_geom.intersection(slice_bounds_geom)
                        if clipped_geom.is_empty: 
                            continue
                        if (clipped_geom.area / t_geom.area) < args.min_visible_ratio: 
                            continue
                            
                        # Convert to YOLO format
                        yolo_coords = shapely_to_yolo_polygon(clipped_geom, window_transform, slice_size)
                        if yolo_coords: 
                            yolo_labels.append(f"{t_class} {yolo_coords}")
                            
                    # Save output if it contains valid labels
                    if len(yolo_labels) > 0:
                        base_name = f"real_{split_name}_tree{idx}_aug{aug_idx}"
                        cv2.imwrite(os.path.join(args.output_dir, 'images', split_name, f"{base_name}.jpg"), bg_slice)
                        with open(os.path.join(args.output_dir, 'labels', split_name, f"{base_name}.txt"), "w") as f:
                            f.write("\n".join(yolo_labels))
                        slice_counter += 1
                        
    return slice_counter

def main():
    args = parse_args()
    
    print(f"Initializing YOLO Dataset Pipeline for target: {args.target_species} (Column: {args.species_column})")
    
    # Create standard YOLO directory structure
    for split in ['train', 'val']:
        os.makedirs(os.path.join(args.output_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, 'labels', split), exist_ok=True)
        
    print("Loading shapefile and splitting Train/Val datasets...")
    gdf_all = gpd.read_file(args.shp_path)
    
    gdf_target = gdf_all[gdf_all[args.species_column] == args.target_species].copy()
    gdf_others = gdf_all[gdf_all[args.species_column] != args.target_species].copy()
    
    train_target, val_target = train_test_split(gdf_target, test_size=args.val_split, random_state=42)
    train_others, val_others = train_test_split(gdf_others, test_size=args.val_split, random_state=42)
    
    train_all = gpd.pd.concat([train_target, train_others])
    val_all = gpd.pd.concat([val_target, val_others])
    
    tif_files = glob.glob(os.path.join(args.tif_dir, "**", "*.tif"), recursive=True)
    
    if not tif_files:
        print(f"Error: No TIF files found in {args.tif_dir}")
        return

    print("\n[Stage 1/2] Generating [Validation Set] (No augmentation, 1:1 sampling)...")
    val_c1 = generate_real_dataset(val_target, val_all, tif_files, 'val', args, augments_per_tree=1)
    val_c2 = generate_real_dataset(val_others, val_all, tif_files, 'val', args, augments_per_tree=1)
    print(f"Validation set generated successfully. Total slices: {val_c1 + val_c2}")

    print(f"\n[Stage 2/2] Generating [Training Set] ({args.train_target_aug}x augmentation for targets)...")
    train_c1 = generate_real_dataset(train_target, train_all, tif_files, 'train', args, augments_per_tree=args.train_target_aug)
    train_c2 = generate_real_dataset(train_others, train_all, tif_files, 'train', args, augments_per_tree=args.train_bg_aug)
    print(f"Training set generated successfully. Total slices: {train_c1 + train_c2}")
    
    print(f"\nPipeline completed! Dataset saved to: {args.output_dir}")

if __name__ == "__main__":
    main()