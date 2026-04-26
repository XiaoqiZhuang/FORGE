import os
import shutil
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import joblib

# ==========================================
# ⚙️ 配置区
# ==========================================
DATA_ROOT = "X:/ai4eo/Shared/2025_Forge/Tree_Features"
CSV_PATH = os.path.join(DATA_ROOT, "annotations_cleaned.csv")
IMAGE_DIR = os.path.join(DATA_ROOT, "images")
EMBEDDINGS_PATH = os.path.join(DATA_ROOT, "embeddings_final.npy")  
TARGET_CLASS_ID = 10


def analyze_and_save():
    # 1. 加载对齐后的数据
    X = np.load(EMBEDDINGS_PATH)
    df = pd.read_csv(CSV_PATH)
    y = df['class_id'].apply(lambda x: 1 if x == TARGET_CLASS_ID else 0).values

    # 获取原始索引以便追踪 ID
    indices = np.arange(len(y))

    # 2. 划分测试集（必须与你之前验证时的 random_state 一致）
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, indices, test_size=0.2, random_state=42, stratify=y
    )

    # 3. 训练并预测
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = SVC(kernel='rbf', class_weight='balanced', probability=True)
    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)


    # 1. 保存训练好的 SVM 模型
    svm_model_path = "X:/ai4eo/Shared/2025_Forge/Tree_Features/best_svm_model.pkl"
    joblib.dump(clf, svm_model_path)
    print(f"✅ SVM 模型已成功保存至: {svm_model_path}")

    # 2. 【极其重要】保存你用来缩放特征的 Scaler
    svm_scaler_path = "X:/ai4eo/Shared/2025_Forge/Tree_Features/svm_scaler.pkl"
    joblib.dump(scaler, svm_scaler_path)
    print(f"✅ Scaler 已成功保存至: {svm_scaler_path}")


if __name__ == "__main__":
    analyze_and_save()