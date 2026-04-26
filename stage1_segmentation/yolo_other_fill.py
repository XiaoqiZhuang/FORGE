import os
import re
import glob
import random
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from shapely.geometry import box, Polygon, Point
from PIL import Image

# ==========================================
# ⚙️ 配置区
# ==========================================
BASE_DIR = r"X:/ai4eo/Shared/2025_Forge/OSINFOR_data/2023/" 
GT_SHP_PATH = r"X:/ai4eo/Shared/2025_Forge/OSINFOR_data/copas-2023/copas_2023_condatos_vs2.shp"
OUTPUT_DIR = r"Z:/mnt/parscratch/users/acr23xz/Forge/yolo_SH_v2"

# 根据你刚才的统计，我们需要补齐的目标（SH * 2）
TARGET_OTHERS = {
    "2023-03": 60,   # 原有11，需补49
    "2023-05": 336,  # 原有62，需补274
    "2023-06": 50,   # 原有8， 需补42
    "2023-08": 300,  # 原有31，需补269
    "2023-10": 354,  # 原有110，需补244
    "2023-11": 412   # 原有148，需补264
}

TILE_SIZE = 1024
CLASS_MAP = {"Shihuahuaco": 0, "Others": 1}

# ==========================================
# 辅助函数
# ==========================================

def get_month_from_name(filename):
    match = re.search(r'_(\d{8})_', filename)
    if match: return f"{match.group(1)[4:]}-{match.group(1)[2:4]}"
    return "Unknown"

def geo_to_yolo(poly, transform, tile_size):
    coords = []
    inv_transform = ~transform
    for x, y in poly.exterior.coords:
        px, py = inv_transform * (x, y)
        coords.append(f"{max(0, min(1, px/tile_size)):.6f} {max(0, min(1, py/tile_size)):.6f}")
    return " ".join(coords)

def process_and_save_others(src, center_point, tile_size, local_gt, img_id):
    transform = src.transform
    half_size_geo = (tile_size / 2) * abs(transform.a)
    tile_geom = box(center_point.x - half_size_geo, center_point.y - half_size_geo, 
                    center_point.x + half_size_geo, center_point.y + half_size_geo)
    try:
        out_image, out_transform = mask(src, [tile_geom], crop=True, all_touched=True)
        if out_image.shape[1] < tile_size or out_image.shape[2] < tile_size: return False
    except: return False

    intersected_trees = local_gt[local_gt.intersects(tile_geom)].copy()
    label_lines = []
    for _, tree in intersected_trees.iterrows():
        clipped_poly = tree.geometry.intersection(tile_geom)
        if clipped_poly.is_empty or not isinstance(clipped_poly, Polygon): continue
        if clipped_poly.area / tree.geometry.area < 0.7: continue
            
        cls_id = CLASS_MAP.get(tree["NOMBRE_COM"], 1)
        yolo_poly = geo_to_yolo(clipped_poly, out_transform, tile_size)
        label_lines.append(f"{cls_id} {yolo_poly}")

    if not label_lines: return False

    Image.fromarray(np.transpose(out_image[:3], (1, 2, 0)).astype('uint8')).save(
        os.path.join(OUTPUT_DIR, "images", f"{img_id}.png"))
    with open(os.path.join(OUTPUT_DIR, "labels", f"{img_id}.txt"), "w") as f:
        f.write("\n".join(label_lines))
    return True

# ==========================================
# 执行逻辑
# ==========================================

def refill_others():
    print("🛠️ 开始 Others 专项补齐任务...")
    gt_gdf = gpd.read_file(GT_SHP_PATH)
    tif_paths = sorted(glob.glob(os.path.join(BASE_DIR, "**", "*.tif"), recursive=True))
    
    # 统计现有数量，避免重复
    existing_files = os.listdir(os.path.join(OUTPUT_DIR, "images"))
    current_counts = {m: 0 for m in TARGET_OTHERS.keys()}
    for f in existing_files:
        if f.startswith("OT_"):
            m = f.split('_')[1]
            if m in current_counts: current_counts[m] += 1

    for path in tif_paths:
        name = os.path.basename(path)
        month = get_month_from_name(name)
        if month not in TARGET_OTHERS: continue
        
        # 检查该月是否已达标
        if current_counts[month] >= TARGET_OTHERS[month]:
            continue

        try:
            with rasterio.open(path) as src:
                local_gt = gt_gdf.to_crs(src.crs)
                local_gt = local_gt[local_gt.intersects(box(*src.bounds))].copy()
                
                # 只选 Others
                ot_list = local_gt[local_gt["NOMBRE_COM"] != "Shihuahuaco"]
                if ot_list.empty: continue
                
                # 计算还需补多少
                needed = TARGET_OTHERS[month] - current_counts[month]
                # 每次从这张图里随机抽一些，不要全切（增加多样性）
                to_sample = min(len(ot_list), needed, 20) 
                
                sampled_ot = ot_list.sample(n=to_sample)
                print(f"📅 补齐中: {name} ({month}) | 计划补齐: {to_sample}")

                for idx, row in sampled_ot.iterrows():
                    img_id = f"OT_REFILL_{month}_{idx}"
                    # 检查是否已存在
                    if f"{img_id}.png" in existing_files: continue
                    
                    if process_and_save_others(src, row.geometry.centroid, TILE_SIZE, local_gt, img_id):
                        current_counts[month] += 1
                        
        except Exception as e:
            print(f"❌ 跳过损坏文件 {name}: {e}")
            continue

    print("\n✅ 补齐任务结束！请再次运行统计脚本检查比例。")

if __name__ == "__main__":
    refill_others()