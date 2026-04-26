import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# ==========================================
# ⚙️ 配置区
# ==========================================
VAL_FOLDERS = [
    "X:/ai4eo/Shared/2025_Forge/2025/2025_11_features",
    "X:/ai4eo/Shared/2025_Forge/2025/2025_12_features"
]
# 缓存路径：下次运行会直接加载它
CACHE_PATH = "X:/ai4eo/Shared/2025_Forge/2025/embeddings_2025_cache.npy"
MODEL_PATH = "X:/ai4eo/Shared/2025_Forge/Tree_Features/best_tree_classifier_oversampled.pth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# 🛠️ 核心提取函数 (带缓存逻辑)
# ==========================================
def get_or_create_embeddings():
    if os.path.exists(CACHE_PATH):
        print(f"🚀 发现缓存，正在加载: {CACHE_PATH}")
        return np.load(CACHE_PATH)

    print("⏳ 未发现缓存，开始提取 2025 数据的 DINO 特征...")
    # 确保加载的模型与训练时一致 (你提取出1024维说明用的是 vitl14)
    extractor = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14').to(DEVICE)
    extractor.eval()

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    all_feats = []
    for folder in VAL_FOLDERS:
        print(f"📂 处理文件夹: {folder}")
        files = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for f in tqdm(files):
            img_path = os.path.join(folder, f)
            try:
                with Image.open(img_path).convert('RGB') as img:
                    img_t = transform(img).unsqueeze(0).to(DEVICE)
                    with torch.no_grad():
                        feat = extractor(img_t).cpu().numpy()[0]
                    all_feats.append(feat)
            except Exception as e:
                print(f"跳过损坏图片 {f}: {e}")

    final_embeddings = np.array(all_feats)
    np.save(CACHE_PATH, final_embeddings)
    print(f"✅ 特征提取完成并保存至缓存。总数: {len(final_embeddings)}")
    return final_embeddings

# ==========================================
# 🧠 验证逻辑
# ==========================================
def validate_2025():
    # 1. 载入模型
    checkpoint = torch.load(MODEL_PATH, weights_only=False)
    state_dict = checkpoint['model_state_dict']
    
    # 自动识别模型权重里的输入维度
    expected_dim = state_dict['net.0.weight'].shape[1]
    print(f"📦 模型期望输入维度: {expected_dim}")

    # 2. 获取数据 (加载缓存或实时提取)
    X = get_or_create_embeddings()
    
    # 3. 维度强制对齐 (防止 1026 与 1024 的冲突)
    # 如果你的 X 是 1026 维但模型要 1024，就切片；反之亦然。
    if X.shape[1] != expected_dim:
        print(f"⚠️ 维度不匹配: 数据({X.shape[1]}) vs 模型({expected_dim})，正在自动对齐...")
        X = X[:, :expected_dim]

    # 4. 推理
    # 动态重建模型类 (确保结构一致)
    class TreeClassifier(torch.nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(input_dim, 256),
                torch.nn.BatchNorm1d(256),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.4),
                torch.nn.Linear(256, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 2)
            )
        def forward(self, x): return self.net(x)

    model = TreeClassifier(expected_dim).to(DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        X_tensor = torch.FloatTensor(X).to(DEVICE)
        outputs = model(X_tensor)
        preds = torch.max(outputs, 1)[1].cpu().numpy()
        probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()

    # 5. 报告
    total = len(preds)
    sh_count = np.sum(preds == 1)
    print("\n" + "="*40)
    print("📊 2025 Shihuahuaco 专项验证报告")
    print("="*40)
    print(f"样本总数: {total}")
    print(f"成功识别 (Recall): {sh_count} ({sh_count/total:.2%})")
    print(f"漏报数量: {total - sh_count}")
    print(f"平均置信度: {probs.mean():.4f}")
    print("="*40)

if __name__ == "__main__":
    validate_2025()