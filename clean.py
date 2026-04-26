

DATA_ROOT = "X:/ai4eo/Shared/2025_Forge/Tree_Features"

import os
import numpy as np
import pandas as pd
from tqdm import tqdm

# ==========================================
# ⚙️ 配置区
# ==========================================
IMAGE_DIR = os.path.join(DATA_ROOT, "images")
OLD_CSV = os.path.join(DATA_ROOT, "annotations.csv")
CLEANED_CSV = os.path.join(DATA_ROOT, "annotations_cleaned.csv")
OLD_EMBEDDINGS = os.path.join(DATA_ROOT, "embeddings.npy")

# 输出文件
FINAL_EMBEDDINGS = os.path.join(DATA_ROOT, "embeddings_final.npy")

def sync_and_clean():
    # 1. 加载所有数据
    print("📅 正在加载原始数据...")
    df_old = pd.read_csv(OLD_CSV)
    df_cleaned = pd.read_csv(CLEANED_CSV)
    X_old = np.load(OLD_EMBEDDINGS)

    if len(df_old) != len(X_old):
        print(f"❌ 严重错误：原始 CSV ({len(df_old)}) 与 Embeddings ({len(X_old)}) 长度不符！请检查是否已经手动改动过。")
        return

    # 2. 识别需要保留和删除的 ID
    kept_ids = set(df_cleaned['id'].tolist())
    all_ids = df_old['id'].tolist()
    
    # 3. 物理删除磁盘上的图片文件 (仅删除不在 cleaned 列表中的)
    print("🗑️ 正在清理物理图片文件...")
    removed_files = 0
    for img_id in tqdm(all_ids):
        if img_id not in kept_ids:
            # 搜索并删除所有可能的后缀
            for ext in ['.png', '.jpg', '.jpeg', '.tif']:
                p = os.path.join(IMAGE_DIR, f"{img_id}{ext}")
                if os.path.exists(p):
                    os.remove(p)
                    removed_files += 1
    
    # 4. 同步切片特征矩阵
    print("✂️ 正在重新切片特征矩阵 (Embedding Matrix)...")
    # 找到 cleaned 数据在原数据中的索引位置
    df_old['original_index'] = range(len(df_old))
    kept_indices = df_old[df_old['id'].isin(kept_ids)]['original_index'].values
    
    # 提取对应的特征行
    X_final = X_old[kept_indices]
    
    # 验证最终长度
    if len(X_final) == len(df_cleaned):
        np.save(FINAL_EMBEDDINGS, X_final)
        print("\n" + "="*40)
        print("✅ 同步清洗完成！")
        print("="*40)
        print(f"📊 最终样本数: {len(X_final)}")
        print(f"🖼️ 物理删除图片数: {removed_files}")
        print(f"💾 新特征矩阵已保存至: {FINAL_EMBEDDINGS}")
        print("="*40)
        print("💡 现在你可以使用新的 annotations_cleaned.csv 和 embeddings_final.npy 运行之前的分析脚本了。")
    else:
        print(f"❌ 逻辑错误：最终矩阵长度 ({len(X_final)}) 与 清洗后CSV ({len(df_cleaned)}) 不匹配！")

if __name__ == "__main__":
    # 安全确认
    confirm = input("⚠️ 该操作将从磁盘删除图片文件并生成新特征矩阵，确定继续吗？(y/n): ")
    if confirm.lower() == 'y':
        sync_and_clean()
    else:
        print("操作已取消。")