import os
import random
import shutil
from pathlib import Path

# 1. 設定原始路徑 (WSL 路徑)
root_dir = "/mnt/e/plant/MAE_plant/PlantCLEF2024singleplanttrainingdata_800_max_side_size/images_max_side_800"
target_dir = root_dir # 我們直接在原目錄下建立 train/val/test

def split_dataset():
    # 建立目標資料夾
    for s in ['train', 'val', 'test']:
        os.makedirs(os.path.join(target_dir, s), exist_ok=True)

    # 取得所有植物類別資料夾 (排除剛剛建立的 train/val/test)
    all_classes = [d for d in os.listdir(root_dir) 
                   if os.path.isdir(os.path.join(root_dir, d)) and d not in ['train', 'val', 'test']]

    print(f"找到 {len(all_classes)} 個類別，開始切分...")

    for i, cls in enumerate(all_classes):
        src_cls_path = os.path.join(root_dir, cls)
        all_imgs = [f for f in os.listdir(src_cls_path) if os.path.isfile(os.path.join(src_cls_path, f))]
        
        # 打亂圖片順序
        random.shuffle(all_imgs)

        # 計算切分索引
        n = len(all_imgs)
        train_idx = int(n * 0.7)
        val_idx = int(n * 0.9)

        splits = {
            'train': all_imgs[:train_idx],
            'val': all_imgs[train_idx:val_idx],
            'test': all_imgs[val_idx:]
        }

        # 執行移動 (shutil.move 在同一個磁碟下幾乎瞬發)
        for s_name, imgs in splits.items():
            if len(imgs) > 0:
                dst_path = os.path.join(target_dir, s_name, cls)
                os.makedirs(dst_path, exist_ok=True)
                for img in imgs:
                    shutil.move(os.path.join(src_cls_path, img), os.path.join(dst_path, img))
        
        if (i + 1) % 100 == 0:
            print(f"已處理 {i + 1} / {len(all_classes)} 個類別...")

    # 刪除原本空的類別資料夾
    print("清理空資料夾...")
    for cls in all_classes:
        try:
            os.rmdir(os.path.join(root_dir, cls))
        except:
            pass

    print("✅ 721 切分完成！")

if __name__ == "__main__":
    split_dataset()
