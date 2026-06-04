import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim

# 🌟 替換為學術界最通用的 pytorch-fid 庫
from pytorch_fid.fid_score import calculate_fid_given_paths

# 🌟 請確保你的 Mamba 類別與 ViT 類別都寫在同目錄下的 models_mae.py 裡
import models_mae 

'''
====================================================================
🏆 MAE 雙軌驗證品質評估腳本 (防灌水標準版)
====================================================================
'''

def unpatchify(x, p=16):
    """
    獨立的影像還原函式，不再依賴模型內部的屬性。
    x: (N, L, patch_size**2 * 3) -> [B, 196, 768]
    回傳 imgs: (N, 3, H, W) -> [B, 3, 224, 224]
    """
    h = w = int(x.shape[1] ** 0.5)
    assert h * w == x.shape[1], "Token 數量與 2D 網格大小不匹配！"
    
    x = x.reshape(shape=[x.shape[0], h, w, p, p, 3])
    x = torch.einsum('nhwpqc->nchpwq', x)
    imgs = x.reshape(shape=[x.shape[0], 3, h * p, w * p])
    return imgs

def get_args():
    parser = argparse.ArgumentParser(description='MAE Reconstruction Evaluation')
    parser.add_argument('--model', default='mae_mamba_vision_tiny', type=str, 
                        help='模型名稱 (例如: mae_mamba_vision_tiny 或 mae_vit_base_patch16)')
    parser.add_argument('--checkpoint', required=True, type=str, 
                        help='預訓練權重路徑 (.pth)')
    parser.add_argument('--data_path', required=True, type=str, 
                        help='測試集圖片路徑 (指向 test 資料夾)')
    parser.add_argument('--device', default='cuda', type=str, help='運算裝置')
    parser.add_argument('--mask_ratio', default=0.75, type=float, help='預訓練遮罩比例')
    parser.add_argument('--output_dir', default='./eval_results', type=str, 
                        help='暫存真實與還原圖片的總資料夾')
    return parser.parse_args()

def main():
    args = get_args()
    device = torch.device(args.device) if torch.cuda.is_available() else torch.device('cpu')
    
    # 🌟 【PyTorch 2.6+ 安全更新】
    torch.serialization.add_safe_globals([argparse.Namespace])
    
    # 強制將路徑轉為絕對路徑
    abs_output_dir = os.path.abspath(args.output_dir)
    real_dir = os.path.join(abs_output_dir, args.model, 'real')
    pred_dir = os.path.join(abs_output_dir, args.model, 'pred')
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)

    # --------------------------------------------------------------------------
    # 1. 載入模型與 Pre-trained 權重
    # --------------------------------------------------------------------------
    print(f" ⚙️  正在從 models_mae 初始化模型架構: {args.model}...")
    if args.model not in models_mae.__dict__:
        raise KeyError(f"❌ 在 models_mae 找不到名為 '{args.model}' 的註冊函式！")
        
    model = models_mae.__dict__[args.model]()
    model.to(device)
    
    print(f" 📂 正在載入權重檔案: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    
    if 'model' in checkpoint:
        msg = model.load_state_dict(checkpoint['model'], strict=False)
    else:
        msg = model.load_state_dict(checkpoint, strict=False)
    print(f" 💡 權重載入狀態回報：{msg}")
    model.eval()

    # --------------------------------------------------------------------------
    # 2. 準備測試集與反正規化轉換
    # --------------------------------------------------------------------------
    transform = transforms.Compose([
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )

    dataset = datasets.ImageFolder(args.data_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    # 軌道一：傳統拼接指標 (容易對齊方塊而灌水)
    ssim_recon_list = []
    psnr_recon_list = []
    
    # 軌道二：純模型預測指標 (真實反映模型脫離遮罩後的重建功力)
    ssim_pure_list = []
    psnr_pure_list = []
    
    print(f" 🚀 成功讀取測試集！開始處理 {len(dataset)} 張圖片並進行雙軌計算...")
    p = 16

    # --------------------------------------------------------------------------
    # 3. 核心評估迴圈
    # --------------------------------------------------------------------------
    with torch.no_grad():
        for idx, (imgs, _) in enumerate(dataloader):
            imgs = imgs.to(device)
            
            # Forward 得到預測值與遮罩
            loss, pred, mask = model(imgs, mask_ratio=args.mask_ratio)
            
            # 透過 unpatchify 還原模型生成的整張影像
            pred_imgs = unpatchify(pred, p=p)
            
            # 建立 2D 遮罩
            mask_2d = mask.detach().unsqueeze(-1).repeat(1, 1, p**2 * 3)
            mask_2d = unpatchify(mask_2d, p=p)
            
            # 【影像 A】正統 MAE 拼接圖：保留區抓原圖，遮罩區抓預測圖
            reconstructed_imgs = imgs * (1 - mask_2d) + pred_imgs * mask_2d
            
            # --- 轉回 0~255 的 HWC Numpy 格式 ---
            real_img_np = (inv_normalize(imgs[0]).clamp(0, 1).cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
            recon_img_np = (inv_normalize(reconstructed_imgs[0]).clamp(0, 1).cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
            
            # 【影像 B】純預測圖：不拼接，看模型自己畫的整張 224x224
            pure_pred_img_np = (inv_normalize(pred_imgs[0]).clamp(0, 1).cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
            
            # 儲存實體圖片 (注意：FID 計算學術標準一律使用【拼接圖】)
            Image.fromarray(real_img_np).save(os.path.join(real_dir, f"{idx}.png"))
            Image.fromarray(recon_img_np).save(os.path.join(pred_dir, f"{idx}.png"))
            
            # 計算兩軌指標
            ssim_recon = compute_ssim(real_img_np, recon_img_np, channel_axis=2)
            psnr_recon = compute_psnr(real_img_np, recon_img_np)
            
            ssim_pure = compute_ssim(real_img_np, pure_pred_img_np, channel_axis=2)
            psnr_pure = compute_psnr(real_img_np, pure_pred_img_np)
            
            ssim_recon_list.append(ssim_recon)
            psnr_recon_list.append(psnr_recon)
            ssim_pure_list.append(ssim_pure)
            psnr_pure_list.append(psnr_pure)
            
            if (idx + 1) % 100 == 0:
                print(f"    [進度 {idx+1}/{len(dataset)}]")
                print(f"    └─ 拼接軌道平均 -> SSIM: {np.mean(ssim_recon_list):.4f}, PSNR: {np.mean(psnr_recon_list):.2f} dB")
                print(f"    └─ 純預測軌道平均 -> SSIM: {np.mean(ssim_pure_list):.4f}, PSNR: {np.mean(psnr_pure_list):.2f} dB")

    # --------------------------------------------------------------------------
    # 4. 計算全局感知特徵距離 (FID)
    # --------------------------------------------------------------------------
    print("\n 🔥 正在呼叫 Inception-V3 網路計算學術標準 FID...")
    try:
        fid_val = calculate_fid_given_paths(
            paths=[real_dir, pred_dir],
            batch_size=50,
            device=device,
            dims=2048,
            num_workers=4
        )
    except Exception as e:
        print(f"⚠️ FID 計算遇到問題: {e}")
        fid_val = float('nan')

    # --------------------------------------------------------------------------
    # 5. 列印學術對決成果報告
    # --------------------------------------------------------------------------
    print("\n" + "="*65)
    print(f"🏆  MAE 雙軌評估驗證報告 ({args.model}) 🏆")
    print("="*65)
    print(f"📊 測試影像總數 : {len(dataset)} 張")
    print("-"*65)
    print("【 軌道一：傳統拼接圖評估 (包含 25% 原圖保留區) 】")
    print(f"🔺 平均 SSIM (Recon)   : {np.mean(ssim_recon_list):.4f}")
    print(f"🔺 平均 PSNR (Recon)   : {np.mean(psnr_recon_list):.2f} dB")
    print("-"*65)
    print("【 🌟 軌道二：純模型預測評估 (100% 檢驗模型自主生成能力) 】")
    print(f"🔺 平均 SSIM (Pure)    : {np.mean(ssim_pure_list):.4f}")
    print(f"🔺 平均 PSNR (Pure)    : {np.mean(psnr_pure_list):.2f} dB")
    print("-"*65)
    print(f"🔻 學術標準 FID (基於拼接圖) : {fid_val:.4f}" if not np.isnan(fid_val) else "🔻 最終 FID : 計算失敗")
    print("="*65)
    print(f"📂 評估用實體圖片保存在: {pred_dir}\n")

if __name__ == '__main__':
    main()
