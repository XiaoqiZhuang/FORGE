import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

# ==========================================
# ⚙️ 配置区
# ==========================================
DATA_ROOT = "X:/ai4eo/Shared/2025_Forge/Tree_Features"
CSV_PATH = os.path.join(DATA_ROOT, "annotations_cleaned.csv")
IMAGE_DIR = os.path.join(DATA_ROOT, "images")
EMBEDDINGS_PATH = os.path.join(DATA_ROOT, "embeddings_final.npy") 
SAVE_PATH = os.path.join(DATA_ROOT, "best_tree_classifier_oversampled.pth")

TARGET_CLASS_ID = 10
USE_EXTRA_FEATURES = False  
BG_COLOR = np.array([124, 116, 104])

# 🚀 训练超参数
BATCH_SIZE = 32
EPOCHS = 100
LR = 5e-5  # 稍低的学习率，配合过采样防止过拟合

# ==========================================
# 🛠️ 特征提取逻辑 (保持不变)
# ==========================================
def extract_physical_features(df, image_dir):
    print(f"\n📏 开始提取物理特征 (目标: {len(df)} 张图像)...")
    print(f"🎨 排除背景色: {BG_COLOR}")
    start_time = time.time()
    
    physical_feats = []
    total = len(df)
    ids = df['id'].tolist()
    
    for i, img_id in enumerate(ids):
        # 每处理 100 张打印一次，且跳过第 0 张以防除零
        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            # 加上 1e-6 防止 ZeroDivisionError
            speed = (i + 1) / (elapsed + 1e-6)
            remaining = (total - (i + 1)) / speed
            print(f"⏳ 进度: {i+1}/{total} | 速度: {speed:.2f}张/秒 | 预计剩余时间: {remaining:.1f}秒")

        img_path = os.path.join(image_dir, f"{img_id}.png")
        if not os.path.exists(img_path):
            img_path = os.path.join(image_dir, f"{img_id}.jpg")
            
        try:
            with Image.open(img_path).convert('RGB') as img:
                img_arr = np.array(img)
                # 向量化计算像素差值
                diff = np.sum(np.abs(img_arr - BG_COLOR), axis=-1)
                area = np.sum(diff > 20) 
                w, h = img.size
                physical_feats.append([float(area), float(w/h)])
        except Exception:
            physical_feats.append([0.0, 1.0])
            
    print(f"✅ 特征提取完成！总耗时: {time.time() - start_time:.1f}秒\n")
    return np.array(physical_feats, dtype=np.float32)

# ==========================================
# 🧠 模型结构 (轻量化以适应小样本)
# ==========================================
class TreeClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )
    def forward(self, x): return self.net(x)

def train_eval():
    # 1. 加载数据
    X_dino = np.load(EMBEDDINGS_PATH)
    df = pd.read_csv(CSV_PATH)
    y = df['class_id'].apply(lambda x: 1 if x == TARGET_CLASS_ID else 0).values

    # 2. 特征准备
    if USE_EXTRA_FEATURES:
        X_phys = extract_physical_features(df, IMAGE_DIR)
        X = np.hstack((X_dino, StandardScaler().fit_transform(X_phys)))
    else: X = X_dino

    # 3. 数据集划分
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 🔥 核心：过采样逻辑 (Oversampling)
    class_sample_count = np.array([len(np.where(y_train == t)[0]) for t in np.unique(y_train)])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in y_train])
    samples_weight = torch.from_numpy(samples_weight)
    sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))

    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    # 使用 sampler 后，每一个 batch 中 0 和 1 的比例会接近 1:1
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)

    # 4. 初始化
    model = TreeClassifier(X.shape[1]).to("cuda")
    criterion = nn.CrossEntropyLoss() # 过采样后不需要再设 weight
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_f1 = -1.0
    print(f"\n🚀 开始训练 (过采样模式)... 训练集样本: {len(X_train)}, 测试集样本: {len(X_test)}")
    print("-" * 60)

    for epoch in range(EPOCHS):
        model.train()
        train_loss, correct, total = 0, 0, 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to("cuda"), batch_y.to("cuda")
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += batch_y.size(0)
            correct += predicted.eq(batch_y).sum().item()

        # 验证阶段
        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                test_x = torch.FloatTensor(X_test).to("cuda")
                test_outputs = model(test_x)
                preds = torch.max(test_outputs, 1)[1].cpu().numpy()
                
                report = classification_report(y_test, preds, target_names=['Others', 'SH'], output_dict=True, zero_division=0)
                sh_f1 = report['SH']['f1-score']
                sh_recall = report['SH']['recall']
                
                # 打印清晰的进度
                print(f"Epoch [{epoch+1:3d}/{EPOCHS}] | Loss: {train_loss/len(train_loader):.4f} | "
                      f"SH Recall: {sh_recall:.2f} | SH F1: {sh_f1:.2f} | 预测SH数: {np.sum(preds)}")

                if sh_f1 > best_f1:
                    best_f1 = sh_f1
                    torch.save({'model_state_dict': model.state_dict(), 'input_dim': X.shape[1]}, SAVE_PATH)
                    print(f"   ⭐ 发现更好模型，已保存 (F1: {best_f1:.4f})")

    # 5. 最终总结
    print("-" * 60)
    if os.path.exists(SAVE_PATH):
        checkpoint = torch.load(SAVE_PATH, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        with torch.no_grad():

            final_preds = torch.max(model(torch.FloatTensor(X_test).to("cuda")), 1)[1].cpu().numpy()
            print("\n📊 最佳模型最终性能报告:")
            print(classification_report(y_test, final_preds, target_names=['Others', 'Shihuahuaco']))
            print("混淆矩阵:")
            print(confusion_matrix(y_test, final_preds))
    else:
        print("❌ 训练未能生成有效模型。")

if __name__ == "__main__":
    train_eval()
