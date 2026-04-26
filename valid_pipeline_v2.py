import os
import cv2
import torch
import numpy as np
import joblib
import matplotlib.pyplot as plt
from PIL import Image
from ultralytics import YOLO
from torchvision import transforms
from torch import nn
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.calibration import calibration_curve
from tqdm import tqdm

# ==========================================
# ⚙️ 配置区
# ==========================================
YOLO_MODEL_PATH = "Z:/mnt/parscratch/users/acr23xz/Forge/sh_experiment_v1/weights/best.pt"
CLS_MODEL_PATH = "X:/ai4eo/Shared/2025_Forge/Tree_Features/best_tree_classifier_oversampled.pth"
VALID_DIR = "X:/ai4eo/Shared/2025_Forge/2025/valid"

# 可视化输出主目录与子目录
VIS_OUT_DIR = "X:/ai4eo/Shared/2025_Forge/2025/vis_output_thr015_filter"
DIR_YOLO_MISS = os.path.join(VIS_OUT_DIR, "1_yolo_missed")       
DIR_CLS_FAIL = os.path.join(VIS_OUT_DIR, "2_classifier_failed")  
DIR_CORRECT = os.path.join(VIS_OUT_DIR, "3_correct_pred")        
DIR_BACKGROUND = os.path.join(VIS_OUT_DIR, "4_pure_background")  

for d in [DIR_YOLO_MISS, DIR_CLS_FAIL, DIR_CORRECT, DIR_BACKGROUND]:
    os.makedirs(d, exist_ok=True)

TARGET_CLASS_ID = 0 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

COLOR_GT = (0, 255, 0)      
COLOR_SH = (0, 0, 255)      
COLOR_OTHER = (0, 255, 255) 

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
    
    # 📈 用于保存校准图所需的数据 (只针对 Classifier 的判定)
    calib_y_true = []
    calib_y_prob = []

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
        MARGIN = 15 # 边缘像素缓冲带，可根据图像分辨率调整
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) < 3: continue
                    if int(parts[0]) == TARGET_CLASS_ID:
                        coords = np.array(list(map(float, parts[1:])))
                        xs, ys = coords[0::2], coords[1::2]
                        x1, y1 = int(np.min(xs)*w), int(np.min(ys)*h)
                        x2, y2 = int(np.max(xs)*w), int(np.max(ys)*h)
                        
                        # 🚫 【核心过滤逻辑】：如果框碰到了图像边缘，直接抛弃
                        if x1 < MARGIN or y1 < MARGIN or x2 > (w - MARGIN) or y2 > (h - MARGIN):
                            continue # 跳过这个边缘残缺目标
                            
                        box = [x1, y1, x2, y2]
                        gt_boxes.append(box)
                        cv2.rectangle(vis_img, (box[0], box[1]), (box[2], box[3]), COLOR_GT, 2)

        # 2. YOLO 检测
        results = yolo_model(img_path, conf=0.15, iou=0.6, verbose=False)[0]
        dt_boxes = results.boxes.xyxy.cpu().numpy()
        has_masks = results.masks is not None
        dt_polygons = results.masks.xy if has_masks else []
        
        all_yolo_preds = [] 
        
        # 3. Classifier 精筛
        for idx, box in enumerate(dt_boxes):
            x1, y1, x2, y2 = map(int, box)
            
            # 🚫 【核心过滤逻辑】：如果 YOLO 预测框碰到了图像边缘，同样抛弃
            if x1 < MARGIN or y1 < MARGIN or x2 > (w - MARGIN) or y2 > (h - MARGIN):
                continue
                
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
            
            # 提特征与预测
            img_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            img_t = preprocess(img_pil).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                dino_feat = extractor(img_t).cpu().numpy()[0]
            
            input_tensor = torch.from_numpy(dino_feat).float().unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                out = classifier(input_tensor)
                probs = torch.softmax(out, dim=1)
                sh_prob = probs[0, 1].item()
                # pred_cls = 1 if sh_prob > 0.5 else 0 
                pred_cls = 1 if sh_prob > 0.15 else 0 
            
            all_yolo_preds.append((x1, y1, x2, y2, pred_cls, sh_prob))
            
            # 🧪 【新增逻辑】为校准图收集数据
            # 检查这个 YOLO 框到底框没框中真实的树
            is_true_sh = 0
            for gt in gt_boxes:
                if calculate_iou(gt, (x1, y1, x2, y2)) > 0.3:
                    is_true_sh = 1
                    break
            calib_y_true.append(is_true_sh)
            calib_y_prob.append(sh_prob)

        # 4. 绘图与文件分类保存（同前，此处略去重复注释保留代码）
        sh_count = 0; other_count = 0
        for box in all_yolo_preds:
            x1, y1, x2, y2, pred_cls, conf = box
            if pred_cls == 1: color = COLOR_SH; sh_count += 1
            else: color = COLOR_OTHER; other_count += 1
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)

        image_has_yolo_miss = False; image_has_cls_fail = False; image_has_correct = False
        matched_pred_indices = set()

        for gt in gt_boxes:
            best_iou = 0; best_pred_idx = -1
            for j, pred_box in enumerate(all_yolo_preds):
                iou = calculate_iou(gt, pred_box[:4])
                if iou > best_iou: best_iou = iou; best_pred_idx = j

            if best_iou > 0.3: 
                pred_cls = all_yolo_preds[best_pred_idx][4]
                if pred_cls == 1:
                    image_has_correct = True; all_gt_labels.append(1); all_pred_labels.append(1)
                    matched_pred_indices.add(best_pred_idx)
                else:
                    image_has_cls_fail = True; all_gt_labels.append(1); all_pred_labels.append(0)
            else:
                image_has_yolo_miss = True; all_gt_labels.append(1); all_pred_labels.append(0)

        for j, pred_box in enumerate(all_yolo_preds):
            if pred_box[4] == 1 and j not in matched_pred_indices:
                is_fp = True
                for gt in gt_boxes:
                    if calculate_iou(gt, pred_box[:4]) > 0.3: is_fp = False; break
                if is_fp: all_gt_labels.append(0); all_pred_labels.append(1)

        if len(gt_boxes) == 0 and sh_count == 0:
            all_gt_labels.append(0); all_pred_labels.append(0)

        base_name = f"vis_{img_name}"
        if image_has_yolo_miss: cv2.imwrite(os.path.join(DIR_YOLO_MISS, base_name), vis_img)
        if image_has_cls_fail: cv2.imwrite(os.path.join(DIR_CLS_FAIL, base_name), vis_img)
        if image_has_correct: cv2.imwrite(os.path.join(DIR_CORRECT, base_name), vis_img)
        if not image_has_yolo_miss and not image_has_cls_fail and not image_has_correct:
            cv2.imwrite(os.path.join(DIR_BACKGROUND, base_name), vis_img)

    # ==========================================
    # 📈 绘制并保存 Calibration Curve (可靠性图)
    # ==========================================
    print("\n" + "="*50)
    print("📈 正在生成 Classifier 置信度校准图...")
    
    if len(calib_y_true) > 0:
        # 划分为 10 个区间 (Bins)
        fraction_of_positives, mean_predicted_value = calibration_curve(calib_y_true, calib_y_prob, n_bins=10)
        
        plt.figure(figsize=(10, 10))
        # 上半部分：校准曲线
        ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
        ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated (y=x)")
        ax1.plot(mean_predicted_value, fraction_of_positives, "s-", color="red", label="Classifier")
        ax1.set_ylabel("True Fraction of Positives (Real Accuracy)")
        ax1.set_ylim([-0.05, 1.05])
        ax1.legend(loc="lower right")
        ax1.set_title('Classifier Reliability Diagram (Calibration Curve)')
        ax1.grid(True, linestyle='--', alpha=0.7)

        # 下半部分：概率分布直方图
        ax2 = plt.subplot2grid((3, 1), (2, 0))
        ax2.hist(calib_y_prob, range=(0, 1), bins=10, color="blue", alpha=0.5, edgecolor="black")
        ax2.set_xlabel("Mean Predicted Probability (Confidence)")
        ax2.set_ylabel("Number of Predictions")
        ax2.grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()
        calib_img_path = os.path.join(VIS_OUT_DIR, "calibration_curve.png")
        plt.savefig(calib_img_path, dpi=300)
        print(f"✅ 校准图已保存至: {calib_img_path}")
    else:
        print("⚠️ 没有收集到任何 YOLO 预测框，无法生成校准图。")

    # ==========================================
    # 📊 最终评估报告
    # ==========================================
    print("\n" + "="*50)
    print("🏆 2025 联合验证最终报告 (YOLO + Classifier)")
    print("="*50)
    if len(all_gt_labels) > 0:
        print(classification_report(all_gt_labels, all_pred_labels, target_names=['Others/Background', 'Shihuahuaco'], zero_division=0))
        print("混淆矩阵:")
        print(confusion_matrix(all_gt_labels, all_pred_labels))
    else:
        print("⚠️ 未匹配到任何样本。")

if __name__ == "__main__":
    run_visualization()