import os
import cv2
import torch
import numpy as np
import joblib
from PIL import Image
from ultralytics import YOLO
from torchvision import transforms
from torch import nn
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm

# ==========================================
# ⚙️ 配置区
# ==========================================
YOLO_MODEL_PATH = "Z:/mnt/parscratch/users/acr23xz/Forge/sh_experiment_v1/weights/best.pt"
CLS_MODEL_PATH = "X:/ai4eo/Shared/2025_Forge/Tree_Features/best_tree_classifier_oversampled.pth"
VALID_DIR = "X:/ai4eo/Shared/2025_Forge/2025/valid"

# 可视化输出主目录与子目录
VIS_OUT_DIR = "X:/ai4eo/Shared/2025_Forge/2025/vis_output"
DIR_YOLO_MISS = os.path.join(VIS_OUT_DIR, "1_yolo_missed")       # YOLO 没找出来的
DIR_CLS_FAIL = os.path.join(VIS_OUT_DIR, "2_classifier_failed")  # YOLO 找到了但 Classifier 拒绝的
DIR_CORRECT = os.path.join(VIS_OUT_DIR, "3_correct_pred")        # 成功预测的
DIR_BACKGROUND = os.path.join(VIS_OUT_DIR, "4_pure_background")  # 纯背景且没报错的 (True Negative)

for d in [DIR_YOLO_MISS, DIR_CLS_FAIL, DIR_CORRECT, DIR_BACKGROUND]:
    os.makedirs(d, exist_ok=True)

TARGET_CLASS_ID = 0 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

COLOR_GT = (0, 255, 0)      # 绿色: 真实值
COLOR_SH = (0, 0, 255)      # 红色: 预测为 Shihuahuaco
COLOR_OTHER = (0, 255, 255) # 黄色: 预测为 Others/Background

# ==========================================
# 📐 工具函数：计算 IoU
# ==========================================
def calculate_iou(boxA, boxB):
    xA, yA, xB, yB = max(boxA[0], boxB[0]), max(boxA[1], boxB[1]), min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

# ==========================================
# 🧠 模型结构与加载
# ==========================================
class TreeClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Dropout(0.4), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 2)
        )
    def forward(self, x): return self.net(x)

print("⏳ 正在载入模型...")
yolo_model = YOLO(YOLO_MODEL_PATH)
extractor = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14').to(DEVICE)
extractor.eval()

checkpoint = torch.load(CLS_MODEL_PATH, weights_only=False)
expected_dim = checkpoint['input_dim']
classifier = TreeClassifier(expected_dim).to(DEVICE)
classifier.load_state_dict(checkpoint['model_state_dict'])
classifier.eval()

preprocess = transforms.Compose([
    transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def run_visualization():
    img_dir = os.path.join(VALID_DIR, "images")
    lbl_dir = os.path.join(VALID_DIR, "labels")
    img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg'))]

    print(f"🎨 开始全量验证并分类生成 {len(img_files)} 张图像...")

    all_gt_labels = []
    all_pred_labels = []

    # 加入进度条
    pbar = tqdm(img_files, desc="Processing Images")

    for img_name in pbar:
        img_path = os.path.join(img_dir, img_name)
        label_path = os.path.join(lbl_dir, img_name.rsplit('.', 1)[0] + ".txt")

        img_cv = cv2.imread(img_path)
        if img_cv is None: continue
        h, w, _ = img_cv.shape
        vis_img = img_cv.copy() 

        # 1. 🟢 解析真值 (GT) 并画在图上
        gt_boxes = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) < 3: continue
                    if int(parts[0]) == TARGET_CLASS_ID:
                        coords = np.array(list(map(float, parts[1:])))
                        xs, ys = coords[0::2], coords[1::2]
                        box = [int(np.min(xs)*w), int(np.min(ys)*h), int(np.max(xs)*w), int(np.max(ys)*h)]
                        gt_boxes.append(box)
                        cv2.rectangle(vis_img, (box[0], box[1]), (box[2], box[3]), COLOR_GT, 2)
                        cv2.putText(vis_img, "GT", (box[0], max(0, box[1]-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_GT, 2)

        # 2. YOLO 检测
        results = yolo_model(img_path, verbose=False)[0]
        dt_boxes = results.boxes.xyxy.cpu().numpy()
        has_masks = results.masks is not None
        dt_polygons = results.masks.xy if has_masks else []
        
        all_yolo_preds = [] 
        
        # 3. Classifier 精筛
        for idx, box in enumerate(dt_boxes):
            x1, y1, x2, y2 = map(int, box)
            crop = img_cv[max(0,y1):min(h,y2), max(0,x1):min(w,x2)].copy()
            if crop.size == 0: continue
            
            # 背景灰化
            if has_masks and len(dt_polygons) > idx:
                poly = dt_polygons[idx]
                if len(poly) > 0:
                    poly_crop = poly - np.array([max(0, x1), max(0, y1)])
                    mask = np.zeros(crop.shape[:2], dtype=np.uint8)
                    cv2.fillPoly(mask, [poly_crop.astype(np.int32)], 255)
                    crop[mask == 0] = (104, 116, 124) 
            
            # 提特征
            img_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            img_t = preprocess(img_pil).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                dino_feat = extractor(img_t).cpu().numpy()[0]
            
            input_feat = dino_feat
            
            # 分类预测 (优化了 tensor 转换避免警告)
            input_tensor = torch.from_numpy(input_feat).float().unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                out = classifier(input_tensor)
                probs = torch.softmax(out, dim=1)
                sh_prob = probs[0, 1].item()
                pred_cls = 1 if sh_prob > 0.5 else 0 
            
            all_yolo_preds.append((x1, y1, x2, y2, pred_cls, sh_prob))

        # 4. 将预测画在图上 (红色/黄色)
        sh_count = 0
        other_count = 0
        for box in all_yolo_preds:
            x1, y1, x2, y2, pred_cls, conf = box
            if pred_cls == 1:
                color = COLOR_SH; label_text = f"SH:{conf:.2f}"; sh_count += 1
            else:
                color = COLOR_OTHER; label_text = f"Other:{conf:.2f}"; other_count += 1
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(vis_img, label_text, (x1, min(h, y2+20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # 辅助信息
        cv2.putText(vis_img, f"GT Count: {len(gt_boxes)} (Green)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_GT, 2)
        cv2.putText(vis_img, f"Pred SH: {sh_count} (Red)", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_SH, 2)
        cv2.putText(vis_img, f"Pred Other: {other_count} (Yellow)", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_OTHER, 2)
        
        # ==========================================
        # 🧪 核心逻辑：计算指标与文件分类保存
        # ==========================================
        image_has_yolo_miss = False
        image_has_cls_fail = False
        image_has_correct = False

        matched_pred_indices = set()

        # A. 遍历所有的真值 (GT)，看看它们被谁发现了/遗漏了
        for gt in gt_boxes:
            best_iou = 0
            best_pred_idx = -1
            for j, pred_box in enumerate(all_yolo_preds):
                iou = calculate_iou(gt, pred_box[:4])
                if iou > best_iou:
                    best_iou = iou
                    best_pred_idx = j

            if best_iou > 0.3: # YOLO 找到了
                pred_cls = all_yolo_preds[best_pred_idx][4]
                if pred_cls == 1:
                    # YOLO 找到且 Classifier 认可是 SH -> 真正确 (TP)
                    image_has_correct = True
                    all_gt_labels.append(1); all_pred_labels.append(1)
                    matched_pred_indices.add(best_pred_idx)
                else:
                    # YOLO 找到了但 Classifier 认为是 Other -> 分类器误杀 (FN)
                    image_has_cls_fail = True
                    all_gt_labels.append(1); all_pred_labels.append(0)
            else:
                # YOLO 根本没找到 -> YOLO 漏检 (FN)
                image_has_yolo_miss = True
                all_gt_labels.append(1); all_pred_labels.append(0)

        # B. 寻找误报 (FP)：Classifier 预测是 SH (红框)，但并没有覆盖任何 GT
        for j, pred_box in enumerate(all_yolo_preds):
            if pred_box[4] == 1 and j not in matched_pred_indices:
                # 再次严谨确认它是否跟任何 GT 有交集
                is_fp = True
                for gt in gt_boxes:
                    if calculate_iou(gt, pred_box[:4]) > 0.3:
                        is_fp = False; break
                if is_fp:
                    all_gt_labels.append(0); all_pred_labels.append(1)

        # C. 处理纯背景图 (True Negative)
        if len(gt_boxes) == 0 and sh_count == 0:
            all_gt_labels.append(0); all_pred_labels.append(0)

        # ==========================================
        # 📁 保存图像至对应文件夹
        # ==========================================
        # 注意：一张图如果包含多个目标，可能会同时存入多个文件夹，这正是我们需要的！
        base_name = f"vis_{img_name}"
        if image_has_yolo_miss:
            cv2.imwrite(os.path.join(DIR_YOLO_MISS, base_name), vis_img)
        if image_has_cls_fail:
            cv2.imwrite(os.path.join(DIR_CLS_FAIL, base_name), vis_img)
        if image_has_correct:
            cv2.imwrite(os.path.join(DIR_CORRECT, base_name), vis_img)
        if not image_has_yolo_miss and not image_has_cls_fail and not image_has_correct:
            # 说明这张图里既没有树，模型也没乱报
            cv2.imwrite(os.path.join(DIR_BACKGROUND, base_name), vis_img)

    # ==========================================
    # 📊 最终评估报告
    # ==========================================
    print("\n" + "="*50)
    print("🏆 2025 联合验证最终报告 (YOLO + Classifier)")
    print("="*50)
    if len(all_gt_labels) > 0:
        print(classification_report(all_gt_labels, all_pred_labels, 
                                    target_names=['Others/Background', 'Shihuahuaco'], 
                                    zero_division=0))
        print("混淆矩阵:")
        print(confusion_matrix(all_gt_labels, all_pred_labels))
        print("\n📂 图像已分类保存至：")
        print(f"  - YOLO 漏检: {DIR_YOLO_MISS}")
        print(f"  - Classifier 误杀: {DIR_CLS_FAIL}")
        print(f"  - 正确预测: {DIR_CORRECT}")
    else:
        print("⚠️ 未匹配到任何样本。")

if __name__ == "__main__":
    run_visualization()