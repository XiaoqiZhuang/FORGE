import os
import glob
import re
import pandas as pd
import geopandas as gpd
import rasterio
from shapely.geometry import box

# ==========================================
# ⚙️ 配置区
# ==========================================
# 指向包含多个日期子文件夹的根目录
BASE_DIR = r"X:/ai4eo/Shared/2025_Forge/OSINFOR_data/2023/" 
# 原始全局标注文件
GT_SHP_PATH = r"X:/ai4eo/Shared/2025_Forge/OSINFOR_data/copas-2023/copas_2023_condatos_vs2.shp"

SPECIES_COL = "NOMBRE_COM"
TARGET_SPECIES = "Shihuahuaco"
# ==========================================

def extract_date_from_name(filename):
    """
    从文件名中提取日期，例如 18032023 -> Month: 03
    """
    match = re.search(r'_(\d{8})_', filename)
    if match:
        date_str = match.group(1)
        # DDMMYYYY 结构，取第3-4位作为月份
        month = date_str[2:4]
        year = date_str[4:]
        return f"{year}-{month}"
    return "Unknown"

def analyze_distribution():
    print("🚀 正在加载全局标注文件，这可能需要一点时间...")
    gt_gdf = gpd.read_file(GT_SHP_PATH)
    
    # 获取所有 TIF 文件路径
    # 使用递归模式查找所有子目录下的 .tif
    tif_files = glob.glob(os.path.join(BASE_DIR, "**", "*.tif"), recursive=True)
    print(f"📂 发现 {len(tif_files)} 个 TIF 图像文件。")

    stats_list = []

    for tif_path in tif_files:
        tif_name = os.path.basename(tif_path)
        date_tag = extract_date_from_name(tif_name)
        
        try:
            with rasterio.open(tif_path) as src:
                tif_bounds = box(*src.bounds)
                tif_crs = src.crs
                
                # 统一坐标系并筛选落在当前 TIF 范围内的标注
                local_gt = gt_gdf.to_crs(tif_crs) if gt_gdf.crs != tif_crs else gt_gdf
                mask = local_gt.geometry.intersects(tif_bounds)
                trees_in_tif = local_gt[mask]
                
                if not trees_in_tif.empty:
                    # 统计种类
                    counts = trees_in_tif[SPECIES_COL].value_counts()
                    sh_count = counts.get(TARGET_SPECIES, 0)
                    others_count = len(trees_in_tif) - sh_count
                    
                    stats_list.append({
                        "Month": date_tag,
                        "FileName": tif_name,
                        "Shihuahuaco": sh_count,
                        "Others": others_count
                    })
                    print(f"✅ {tif_name} [{date_tag}]: SH={sh_count}, Others={others_count}")
                else:
                    print(f"⚪ {tif_name} [{date_tag}]: 无标注数据")
                    
        except Exception as e:
            print(f"❌ 无法处理文件 {tif_name}: {e}")

    # 汇总数据
    df = pd.DataFrame(stats_list)
    if df.empty:
        print("😭 未能统计到任何数据，请检查路径和日期格式。")
        return

    # 按月份汇总
    monthly_summary = df.groupby("Month")[["Shihuahuaco", "Others"]].sum()
    monthly_summary["Total_Trees"] = monthly_summary["Shihuahuaco"] + monthly_summary["Others"]
    
    print("\n" + "="*40)
    print("📈 每个月份的标注分布汇总")
    print("="*40)
    print(monthly_summary)
    print("="*40)
    
    # 保存结果方便你后续制定计划
    output_csv = os.path.join(os.path.dirname(GT_SHP_PATH), "monthly_distribution.csv")
    monthly_summary.to_csv(output_csv)
    print(f"📊 汇总结果已保存至: {output_csv}")

if __name__ == "__main__":
    analyze_distribution()