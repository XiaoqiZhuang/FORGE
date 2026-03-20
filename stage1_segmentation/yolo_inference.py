import os
import argparse
import numpy as np
import rasterio
import geopandas as gpd
from shapely.geometry import Polygon
from sahi.predict import get_sliced_prediction
from sahi import AutoDetectionModel

def parse_args():
    parser = argparse.ArgumentParser(description="Run SAHI sliced inference on a large TIF and export SHP/PNG.")
    
    # I/O Paths
    parser.add_argument("--img_path", type=str, required=True,
                        help="Path to the input large TIF orthomosaic.")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the YOLO model weights (.pt).")
    parser.add_argument("--output_dir", type=str, default="inference_outputs",
                        help="Directory to save the resulting SHP and PNG files.")
    
    # SAHI Slicing Parameters
    parser.add_argument("--slice_size", type=int, default=1024,
                        help="Size of the sliding window slice.")
    parser.add_argument("--overlap_ratio", type=float, default=0.4,
                        help="Overlap ratio between slices (0.0 to 1.0).")
    
    # Model Parameters
    parser.add_argument("--conf_threshold", type=float, default=0.25,
                        help="Confidence threshold for YOLO predictions.")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to run inference on (e.g., 'cuda:0' or 'cpu').")
    
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    base_name = os.path.splitext(os.path.basename(args.img_path))[0]
    
    print(f"Loading YOLO model via SAHI from: {args.model_path}")
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path=args.model_path,
        confidence_threshold=args.conf_threshold,
        device=args.device
    )
    
    print(f"Reading spatial metadata from TIF: {args.img_path}")
    with rasterio.open(args.img_path) as src:
        transform = src.transform
        crs = src.crs
        
        # Read the image as a numpy array (RGB) for SAHI
        # Note: If the TIF is massive (e.g., > 10GB), this requires adequate RAM.
        img_array = src.read([1, 2, 3]).transpose(1, 2, 0)
    
    print(f"Starting SAHI sliced inference (Slice: {args.slice_size}, Overlap: {args.overlap_ratio})...")
    result = get_sliced_prediction(
        img_array,
        detection_model,
        slice_height=args.slice_size,
        slice_width=args.slice_size,
        overlap_height_ratio=args.overlap_ratio,
        overlap_width_ratio=args.overlap_ratio,
        postprocess_type="NMM",      
        postprocess_match_metric="IOS",
        postprocess_match_threshold=0.3,
        verbose=0
    )


    # Process Masks to Geographic Polygons
    print("Converting prediction masks to geographic coordinates...")
    geo_features = []
    
    for pred in result.object_prediction_list:
        if pred.mask is None:
            continue
            
        # SAHI stores segmentation as a list of lists: [[x1, y1, x2, y2, ...]]
        for poly_coords in pred.mask.segmentation:
            if len(poly_coords) < 6: # Needs at least 3 points (x, y)
                continue
                
            pts = np.array(poly_coords).reshape(-1, 2)
            
            # Apply Rasterio affine transform to convert pixels to spatial coordinates
            geo_pts = [transform * (pt[0], pt[1]) for pt in pts]
            
            try:
                poly = Polygon(geo_pts)
                if not poly.is_valid:
                    poly = poly.buffer(0)
                
                if poly.area > 0:
                    geo_features.append({
                        "geometry": poly,
                        "class_id": pred.category.id,
                        "class_name": pred.category.name,
                        "score": round(pred.score.value, 4)
                    })
            except Exception:
                continue
                
    # 3. Save to Shapefile
    if geo_features:
        gdf = gpd.GeoDataFrame(geo_features, crs=crs)
        shp_output_path = os.path.join(args.output_dir, f"{base_name}_predictions.shp")
        gdf.to_file(shp_output_path)
        print(f"Successfully saved {len(geo_features)} polygons to: {shp_output_path}")
    else:
        print("No objects detected. Shapefile will not be created.")

if __name__ == "__main__":
    main()