import os
import time
import argparse
import cv2
import numpy as np
from glob import glob
from shapely.geometry import Polygon, box

def parse_args():
    parser = argparse.ArgumentParser(description="Standalone Evaluation Script for Tree Crown Detection Models.")
    
    # Dataset and Model Selection
    parser.add_argument("--dataset_dir", type=str, required=True,
                        help="Path to YOLO dataset directory (images/val, labels/val).")
    parser.add_argument("--eval_model", type=str, required=True, choices=["yolo", "detectree2", "deepforest"],
                        help="Which model to evaluate in the current environment.")
    parser.add_argument("--weights", type=str, default=None,
                        help="Path to the model weights. For deepforest, leave empty to use built-in weights.")
    
    # Parameters
    parser.add_argument("--target_class_id", type=int, default=0,
                        help="Class ID of the target species in YOLO GT.")
    parser.add_argument("--iou_threshold", type=float, default=0.5,
                        help="IoU threshold for matching.")
    parser.add_argument("--conf_threshold", type=float, default=0.1,
                        help="Confidence threshold for predictions.")
    
    return parser.parse_args()

# ==========================================
# Helper Functions for Geometry & Visualization
# ==========================================
def get_valid_polygon(coords):
    if len(coords) < 3:
        return None
    poly = Polygon(coords)
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly

def get_poly_pts(geom):
    pts_list = []
    if geom.geom_type == 'Polygon':
        pts_list.append(np.array(geom.exterior.coords, dtype=np.int32))
    elif geom.geom_type == 'MultiPolygon':
        geoms = getattr(geom, "geoms", geom)
        for p in geoms:
            pts_list.append(np.array(p.exterior.coords, dtype=np.int32))
    return pts_list

def draw_transparent_predictions(img, pred_polys, hit_gts, missed_gts):
    overlay = img.copy()
    output = img.copy()
    
    # 1. Fill Predicted Polygons (Green) on the overlay layer
    for poly in pred_polys:
        for pts in get_poly_pts(poly):
            cv2.fillPoly(overlay, [pts], (0, 255, 0))
            
    # Apply alpha blending for transparency effect
    alpha = 0.4
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
    
    # 2. Draw Hit GTs (Red outlines, thickness 2) on the final output
    for gt_poly in hit_gts:
        for pts in get_poly_pts(gt_poly):
            cv2.polylines(output, [pts], isClosed=True, color=(0, 0, 255), thickness=2)
            
    # 3. Draw Missed GTs (Thick Red outlines + "Missed" label)
    for gt_poly in missed_gts:
        all_pts = []
        for pts in get_poly_pts(gt_poly):
            cv2.polylines(output, [pts], isClosed=True, color=(0, 0, 255), thickness=4)
            all_pts.extend(pts)
            
        if all_pts:
            all_pts = np.array(all_pts)
            top_y_idx = np.argmin(all_pts[:, 1])
            top_point = all_pts[top_y_idx]
            cv2.putText(output, "Missed", (top_point[0] - 10, top_point[1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
            cv2.putText(output, "Missed", (top_point[0] - 10, top_point[1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
    return output

# ==========================================
# Main Evaluation Pipeline
# ==========================================
def main():
    args = parse_args()
    
    # 1. LAZY LOADING
    predictor = None
    if args.eval_model == "yolo":
        print("Loading YOLO environment...")
        from ultralytics import YOLO
        if not args.weights: raise ValueError("YOLO requires --weights argument.")
        predictor = YOLO(args.weights)
        
    elif args.eval_model == "detectree2":
        print("Loading Detectree2 (Detectron2) environment...")
        from detectron2.config import get_cfg
        from detectron2.engine import DefaultPredictor
        from detectron2 import model_zoo
        import torch
        if not args.weights: raise ValueError("Detectree2 requires --weights argument.")
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
        cfg.MODEL.WEIGHTS = args.weights
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.conf_threshold
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        if not torch.cuda.is_available(): cfg.MODEL.DEVICE = "cpu"
        predictor = DefaultPredictor(cfg)
        
    elif args.eval_model == "deepforest":
        print("Loading DeepForest environment...")
        from deepforest import main as df_main
        import torch
        predictor = df_main.deepforest()
        if args.weights:
            predictor.load_state_dict(torch.load(args.weights))
        else:
            print("No weights provided, using DeepForest default release weights.")
            predictor.load_model(model_name="weecology/deepforest-tree", revision="main")

            
    val_images_dir = os.path.join(args.dataset_dir, "images", "val")
    val_labels_dir = os.path.join(args.dataset_dir, "labels", "val")
    
    image_paths = []
    for ext in ['*.jpg', '*.png', '*.tif', '*.jpeg']:
        image_paths.extend(glob(os.path.join(val_images_dir, ext)))
        
    metrics = {'gt_total': 0, 'pred_total': 0, 'recalled': 0, 'useful': 0, 'total_time': 0.0}
    viz_images = []
    VIZ_GRID_SIZE = 512

    print(f"\nStarting {args.eval_model.upper()} evaluation on {len(image_paths)} images...")
    
    for img_path in image_paths:
        img_name = os.path.basename(img_path)
        label_path = os.path.join(val_labels_dir, img_name.rsplit('.', 1)[0] + '.txt')
        if not os.path.exists(label_path): continue
            
        img = cv2.imread(img_path)
        if img is None: continue
        img_h, img_w = img.shape[:2]
        
        # Parse Ground Truth (Only Target Class!)
        gt_polygons = []
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if int(parts[0]) == args.target_class_id:
                    coords = np.array(parts[1:], dtype=float).reshape(-1, 2)
                    coords[:, 0] *= img_w
                    coords[:, 1] *= img_h
                    poly = get_valid_polygon(coords)
                    if poly and poly.area > 0:
                        gt_polygons.append(poly)
                        
        metrics['gt_total'] += len(gt_polygons)
        pred_polygons = []
        
        # --- MODEL SPECIFIC INFERENCE ---
        if args.eval_model == "yolo":
            start_t = time.time()
            res_yolo = predictor(img, verbose=False, conf=args.conf_threshold)[0]
            metrics['total_time'] += (time.time() - start_t)
            
            if res_yolo.masks is not None:
                for pts in res_yolo.masks.xy:
                    poly = get_valid_polygon(pts)
                    if poly and poly.area > 0: 
                        pred_polygons.append(poly)

        elif args.eval_model == "detectree2":
            start_t = time.time()
            res_dt2 = predictor(img)
            metrics['total_time'] += (time.time() - start_t)
            
            instances = res_dt2["instances"].to("cpu")
            if instances.has("pred_masks"):
                masks = instances.pred_masks.numpy()
                for mask in masks:
                    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for contour in contours:
                        if len(contour) >= 3:
                            poly = get_valid_polygon(contour.squeeze(axis=1))
                            if poly and poly.area > 0: pred_polygons.append(poly)

        elif args.eval_model == "deepforest":
            start_t = time.time()
            boxes_df = predictor.predict_image(path=img_path)
            metrics['total_time'] += (time.time() - start_t)
            
            if boxes_df is not None:
                boxes_df = boxes_df[boxes_df['score'] >= args.conf_threshold]
                for _, row in boxes_df.iterrows():
                    poly = box(row['xmin'], row['ymin'], row['xmax'], row['ymax'])
                    pred_polygons.append(poly)

        metrics['pred_total'] += len(pred_polygons)

        # Calculate Metrics (Recall & Purity)
        hit_gts = []
        missed_gts = []
        
        # Target Recall Check
        for gt_poly in gt_polygons:
            is_recalled = False
            for pred_poly in pred_polygons:
                try:
                    iou = gt_poly.intersection(pred_poly).area / gt_poly.union(pred_poly).area
                    if iou >= args.iou_threshold:
                        is_recalled = True
                        break
                except Exception: continue
            if is_recalled:
                metrics['recalled'] += 1
                hit_gts.append(gt_poly)
            else:
                missed_gts.append(gt_poly)
                
        # Candidate Purity Check
        useful_pred_count = 0
        for pred_poly in pred_polygons:
            for gt_poly in gt_polygons:
                try:
                    iou = pred_poly.intersection(gt_poly).area / pred_poly.union(gt_poly).area
                    if iou >= args.iou_threshold:
                        useful_pred_count += 1
                        break
                except Exception: continue
        metrics['useful'] += useful_pred_count

        # Qualitative Visualization
        if len(gt_polygons) > 0 and len(viz_images) < 16:
            viz_img = draw_transparent_predictions(img, pred_polygons, hit_gts, missed_gts)
            cv2.rectangle(viz_img, (0, 0), (250, 40), (0, 0, 0), -1)
            cv2.putText(viz_img, args.eval_model.upper(), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            viz_img_resized = cv2.resize(viz_img, (VIZ_GRID_SIZE, VIZ_GRID_SIZE))
            viz_images.append(viz_img_resized)

    # Save Visualization Grid
    if viz_images:
        print("\nGenerating 4x4 Qualitative Visualization...")
        while len(viz_images) < 16:
            viz_images.append(np.zeros((VIZ_GRID_SIZE, VIZ_GRID_SIZE, 3), dtype=np.uint8))
        rows = [np.hstack(viz_images[i*4:(i+1)*4]) for i in range(4)]
        mosaic = np.vstack(rows)
        out_name = f"eval_viz_{args.eval_model}.jpg"
        cv2.imwrite(out_name, mosaic)
        print(f"Saved qualitative results to: {out_name}")

    # Print Quantitative Report
    recall = (metrics['recalled'] / metrics['gt_total'] * 100) if metrics['gt_total'] > 0 else 0
    purity = (metrics['useful'] / metrics['pred_total'] * 100) if metrics['pred_total'] > 0 else 0
    fps = len(image_paths) / metrics['total_time'] if metrics['total_time'] > 0 else 0
    
    print("\n" + "="*50)
    print(f"Evaluation Report: {args.eval_model.upper()}")
    print("="*50)
    print(f"Total Target Trees in GT   : {metrics['gt_total']}")
    print(f"Total Predicted Polygons   : {metrics['pred_total']}\n")
    print(f"Target Recall              : {recall:.2f}% ({metrics['recalled']} / {metrics['gt_total']})")
    print(f"Candidate Purity           : {purity:.2f}% ({metrics['useful']} / {metrics['pred_total']})")
    print(f"Inference Speed            : {fps:.2f} FPS")
    print("="*50)

if __name__ == "__main__":
    main()