import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import glob
import random
import sys

# 確保能讀取到當前資料夾的 models_mae.py
sys.path.append('.')
import models_mae

# ==========================================
# 1. 路徑設定 (請根據你的資料夾名稱微調)
# ==========================================
dataset_root = '/mnt/c/Users/msp/Downloads/MAE_plant/PlantVillage_Split_721' 
mamba_chkpt = './output_mamba_pretrain/checkpoint-49.pth'
vit_chkpt = './output_vit_pretrain/checkpoint-49.pth'

# 請確認你在 models_mae.py 裡定義的模型名稱
mamba_model_name = 'mae_mamba_base_patch16' 
vit_model_name = 'mae_vit_base_patch16'

# ==========================================
# 2. 工具函式
# ==========================================
def get_model(model_type, chkpt_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = getattr(models_mae, model_type)()
    checkpoint = torch.load(chkpt_path, map_location='cpu')
    
    # 判斷是存放在 'model' 鍵值下還是直接存放
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
        
    model.to(device).eval()
    return model

def run_mae(model, img_tensor):
    device = next(model.parameters()).device
    with torch.no_grad():
        # mask_ratio=0.75 代表遮住 75% 的像素
        loss, pred, mask = model(img_tensor.to(device), mask_ratio=0.75)
        
        # 將 patch 轉回像素空間
        pred = model.unpatchify(pred)
        pred = pred.detach().cpu()
        
        # 處理 mask 方便視覺化 (14x14 -> 224x224)
        mask = mask.detach().cpu().reshape(1, 14, 14).repeat_interleave(16, 1).repeat_interleave(16, 2)
    return pred, mask

# ==========================================
# 3. 主程式
# ==========================================
def main():
    # --- A. 隨機抓圖 ---
    img_list = glob.glob(os.path.join(dataset_root, '**/*.JPG'), recursive=True) + \
               glob.glob(os.path.join(dataset_root, '**/*.jpg'), recursive=True)
    
    if not img_list:
        print(f"❌ 找不到圖片，目前 dataset_root 為: {os.path.abspath(dataset_root)}")
        return

    img_path = random.choice(img_list)
    print(f"📸 選定圖片: {img_path}")

    # --- B. 影像預處理 ---
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img = Image.open(img_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)

    # --- C. 執行還原 ---
    print("⏳ 正在讀取模型並進行還原...")
    try:
        mamba_model = get_model(mamba_model_name, mamba_chkpt)
        mamba_pred, mask = run_mae(mamba_model, img_tensor)
        
        vit_model = get_model(vit_model_name, vit_chkpt)
        vit_pred, _ = run_mae(vit_model, img_tensor)
    except Exception as e:
        print(f"❌ 執行出錯: {e}")
        return

    # --- D. 反正規化與繪圖 ---
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    # 原始圖
    ori = (img_tensor[0] * std + mean).permute(1, 2, 0).clip(0, 1)
    # 遮罩圖 (1 代表被遮住)
    masked = ori * (1 - mask[0].unsqueeze(0).permute(1, 2, 0))
    # Mamba 還原結果
    mamba_res = (mamba_pred[0] * std + mean).permute(1, 2, 0).clip(0, 1)
    # ViT 還原結果
    vit_res = (vit_pred[0] * std + mean).permute(1, 2, 0).clip(0, 1)

    # --- E. 顯示結果 ---
    plt.figure(figsize=(24, 6))
    titles = ["Original", "Masked (75%)", f"MambaVision Recon\n(Loss: 0.3285)", f"ViT-Base Recon\n(Loss: 0.4512)"]
    images = [ori, masked, mamba_res, vit_res]

    for i in range(4):
        plt.subplot(1, 4, i+1)
        plt.imshow(images[i])
        plt.title(titles[i], fontsize=15)
        plt.axis('off')

    save_path = 'final_mae_comparison.png'
    plt.savefig(save_path, bbox_inches='tight')
    print(f"✅ 對比圖已儲存至: {save_path}")

if __name__ == "__main__":
    main()