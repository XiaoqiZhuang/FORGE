import os
import re
import glob
import random
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from shapely.geometry import box, Polygon, Point
from PIL import Image


BASE_DIR = r"X:/ai4eo/Shared/2025_Forge/OSINFOR_data/2023/" 
GT_SHP_PATH = r"X:/ai4eo/Shared/2025_Forge/OSINFOR_data/copas-2023/copas_2023_condatos_vs2.shp"
OUTPUT_DIR = r"Z:/mnt/parscratch/users/acr23xz/Forge/yolo_SH_v2"

# --- 🛰️ 续跑逻辑控制 ---
# 如果要跳过，设置为报错的文件名；如果要从头开始，设为 None
START_FROM_FILE = "25-PUCC-DE-CPC001-13_25102023_001_orto_pv17_idw_transparent_mosaic_group1.tif" 
HAS_REACHED_TARGET = False if START_FROM_FILE else True 

SPECIES_COL = "NOMBRE_COM"
TARGET_SPECIES = "Shihuahuaco"
TILE_SIZE = 1024  

# 物候倍率
AUG_FACTORS = {
    "2023-11": 2, "2023-10": 3, "2023-08": 6, 
    "2023-05": 6, "2023-03": 5, "2023-06": 5
}

CLASS_MAP = {TARGET_SPECIES: 0, "Others": 1}

os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "labels"), exist_ok=True)

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

def process_and_save(src, center_point, tile_size, local_gt, img_id):
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
            
        cls_id = CLASS_MAP.get(tree[SPECIES_COL], 1)
        yolo_poly = geo_to_yolo(clipped_poly, out_transform, tile_size)
        label_lines.append(f"{cls_id} {yolo_poly}")

    if not label_lines: return False

    # 保存图片和标签
    Image.fromarray(np.transpose(out_image[:3], (1, 2, 0)).astype('uint8')).save(
        os.path.join(OUTPUT_DIR, "images", f"{img_id}.png"))
    with open(os.path.join(OUTPUT_DIR, "labels", f"{img_id}.txt"), "w") as f:
        f.write("\n".join(label_lines))
    return True

# ==========================================
# 执行逻辑
# ==========================================

def start_training_export():
    global HAS_REACHED_TARGET
    print("🚀 启动物理平衡切图程序...")
    gt_gdf = gpd.read_file(GT_SHP_PATH)
    # 排序确保续跑逻辑可靠
    tif_paths = sorted(glob.glob(os.path.join(BASE_DIR, "**", "*.tif"), recursive=True))
    
    summary = {m: 0 for m in AUG_FACTORS.keys()}

    for path in tif_paths:
        name = os.path.basename(path)
        
        # --- 续跑逻辑 ---
        if not HAS_REACHED_TARGET:
            if name == START_FROM_FILE:
                print(f"📍 到达指定位置: {name}，开始从此处后续文件运行...")
                HAS_REACHED_TARGET = True
                continue 
            else:
                continue

        month = get_month_from_name(name)
        if month not in AUG_FACTORS: continue
        
        factor = AUG_FACTORS[month]
        
        try:
            with rasterio.open(path) as src:
                res = abs(src.transform.a)
                local_gt = gt_gdf.to_crs(src.crs)
                local_gt = local_gt[local_gt.intersects(box(*src.bounds))].copy()
                
                sh_list = local_gt[local_gt[SPECIES_COL] == TARGET_SPECIES]
                ot_list = local_gt[local_gt[SPECIES_COL] != TARGET_SPECIES]
                
                if sh_list.empty: continue
                print(f"📅 正在处理: {name}")

                # 1. 导出 Shihuahuaco (格式: SH_YYYY-MM_Index_augN)
                for idx, row in sh_list.iterrows():
                    for f in range(factor):
                        # 偏移增强
                        off_x = random.uniform(-150, 150) * res
                        off_y = random.uniform(-150, 150) * res
                        new_center = Point(row.geometry.centroid.x + off_x, row.geometry.centroid.y + off_y)
                        
                        img_id = f"SH_{month}_{idx}_aug{f}"
                        if process_and_save(src, new_center, TILE_SIZE, local_gt, img_id):
                            summary[month] += 1

                # 2. 导出 Others (格式: OT_YYYY-MM_Index)
                if not ot_list.empty:
                    num_to_sample = len(sh_list) * 2
                    sampled_ot = ot_list.sample(n=min(len(ot_list), num_to_sample))
                    for idx, row in sampled_ot.iterrows():
                        img_id = f"OT_{month}_{idx}"
                        process_and_save(src, row.geometry.centroid, TILE_SIZE, local_gt, img_id)
                        
        except Exception as e:
            print(f"❌ 跳过损坏文件 {name}: {e}")
            continue

    print("\n" + "="*40)
    print("📈 导出完成！")

if __name__ == "__main__":
    start_training_export()