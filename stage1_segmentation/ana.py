import os
from collections import Counter

# 指向你刚才生成的训练数据集路径
IMAGE_DIR =r"Z:/mnt/parscratch/users/acr23xz/Forge/yolo_SH_v2/images"

def summarize_dataset():
    if not os.path.exists(IMAGE_DIR):
        print(f"❌ 找不到路径: {IMAGE_DIR}")
        return

    files = [f for f in os.listdir(IMAGE_DIR) if f.endswith('.png')]
    
    # 初始化统计字典
    # 结构: { '2023-10': {'SH': 0, 'OT': 0}, ... }
    stats = {}

    for f in files:
        parts = f.split('_')
        if len(parts) < 3:
            continue
            
        category = parts[0]  # SH 或 OT
        month = parts[1]     # YYYY-MM
        
        if month not in stats:
            stats[month] = {'SH': 0, 'OT': 0}
        
        if category in stats[month]:
            stats[month][category] += 1

    # 打印结果
    print(f"\n{'Month':<12} | {'Shihuahuaco (SH)':<20} | {'Others (OT)':<15} | {'Ratio (1:N)':<10}")
    print("-" * 65)
    
    total_sh = 0
    total_ot = 0
    
    # 排序打印
    for month in sorted(stats.keys()):
        sh = stats[month]['SH']
        ot = stats[month]['OT']
        ratio = round(ot / sh, 2) if sh > 0 else 0
        
        print(f"{month:<12} | {sh:<20} | {ot:<15} | 1:{ratio}")
        
        total_sh += sh
        total_ot += ot

    print("-" * 65)
    print(f"{'TOTAL':<12} | {total_sh:<20} | {total_ot:<15} | 1:{round(total_ot/total_sh, 2) if total_sh > 0 else 0}")

if __name__ == "__main__":
    summarize_dataset()