import os
import cv2
import numpy as np
import argparse
from pathlib import Path
from glob import glob
from ultralytics import YOLO
from shapely.geometry import Polygon

def parse_args():
    parser = argparse.ArgumentParser(description="Calculate Target Candidate Recall (TCR) for YOLO segmentation models.")
    
    # Paths
    parser.add_argument("--dataset_dir", type=str, help="Path to the YOLO dataset directory.")
    parser.add_argument("--model_path", type=str, help="Path to the YOLO model weights (.pt).")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"], help="Dataset split to evaluate (default: val).")
    
    # Thresholds and Class settings
    parser.add_argument("--target_class_id", type=int, default=0,
                        help="Class ID of the target species in YOLO annotations.")
    parser.add_argument("--iou_threshold", type=float, default=0.5,
                        help="IoU threshold to consider a target successfully recalled.")
    parser.add_argument("--conf_threshold", type=float, default=0.1,
                        help="Confidence threshold for YOLO inference (low value recommended for coarse filtering).")
    
    return parser.parse_args()

def get_valid_polygon(coords):
    """Convert coordinates to a Shapely Polygon and fix self-intersections if necessary."""
    if len(coords) < 3:
        return None
    poly = Polygon(coords)
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly

def main():
    args = parse_args()
    
    val_images_dir = os.path.join(args.dataset_dir, "images", args.split)
    val_labels_dir = os.path.join(args.dataset_dir, "labels", args.split)
    
    print(f"Loading YOLO model from: {args.model_path}")
    model = YOLO(args.model_path)
    
    image_paths = []
    for ext in ['*.jpg', '*.png', '*.tif', '*.jpeg']:
        image_paths.extend(glob(os.path.join(val_images_dir, ext)))
        
    if not image_paths:
        print(f"Error: No images found in {val_images_dir}")
        return

    total_target_trees = 0
    recalled_target_trees = 0

    print(f"Scanning dataset split '{args.split}' with {len(image_paths)} images...")
    
    for img_path in image_paths:
        img_name = os.path.basename(img_path)
        label_path = os.path.join(val_labels_dir, img_name.rsplit('.', 1)[0] + '.txt')
        
        if not os.path.exists(label_path):
            continue
            
        img = cv2.imread(img_path)
        if img is None: 
            continue
        img_h, img_w = img.shape[:2]
        
        # Parse ground truth polygons
        gt_polygons = []
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                class_id = int(parts[0])
                
                if class_id == args.target_class_id:
                    coords = np.array(parts[1:], dtype=float).reshape(-1, 2)
                    coords[:, 0] *= img_w
                    coords[:, 1] *= img_h
                    
                    poly = get_valid_polygon(coords)
                    if poly and poly.area > 0:
                        gt_polygons.append(poly)
                        
        if not gt_polygons:
            continue
            
        total_target_trees += len(gt_polygons)

        # Run YOLO inference
        results = model(img_path, verbose=False, conf=args.conf_threshold)
        
        # Extract all predicted polygons ignoring the predicted class
        pred_polygons = []
        if results[0].masks is not None:
            for pts in results[0].masks.xy:
                poly = get_valid_polygon(pts)
                if poly and poly.area > 0:
                    pred_polygons.append(poly)
                    
        # Calculate recall based on spatial overlap (IoU)
        for gt_poly in gt_polygons:
            is_recalled = False
            for pred_poly in pred_polygons:
                try:
                    intersection = gt_poly.intersection(pred_poly).area
                    union = gt_poly.union(pred_poly).area
                    iou = intersection / union if union > 0 else 0
                    
                    if iou >= args.iou_threshold:
                        is_recalled = True
                        break
                except Exception:
                    continue
                    
            if is_recalled:
                recalled_target_trees += 1

    # Print results summary
    print("\n" + "="*50)
    print("Stage 1 Evaluation: Target Candidate Recall (TCR)")
    print("="*50)
    print(f"Total target trees in Ground Truth : {total_target_trees}")
    print(f"Trees successfully segmented       : {recalled_target_trees}")
    
    if total_target_trees > 0:
        tcr = (recalled_target_trees / total_target_trees) * 100
        print(f"Final TCR                          : {tcr:.2f}%")
        print("-" * 50)
        print(f"Note: {tcr:.2f}% of the target trees were successfully isolated")
        print("from the background and added to the candidate pool for Stage 2.")
    else:
        print("Warning: No target trees found in the dataset annotations.")
        print("Please verify the --target_class_id parameter.")

if __name__ == "__main__":
    main()