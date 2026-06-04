import os
import shutil
from datetime import datetime

# 1. 定義時間分界點 (由你的 log.txt 推算)
# Mamba 預訓練結束時間：2026-04-22 22:47
# ViT 預訓練結束時間：2026-04-23 01:24
mamba_end_time = datetime(2026, 4, 22, 22, 47).timestamp()
vit_end_time = datetime(2026, 4, 23, 1, 24).timestamp()

# 2. 設定資料夾路徑
base_dir = './output_dir'
paths = {
    'mamba_pre': './output_dir/pretrain/mamba',
    'vit_pre': './output_dir/pretrain/vit',
    'vit_fine': './output_dir/finetune/vit'
}

# 建立分類資料夾
for p in paths.values():
    os.makedirs(p, exist_ok=True)

# 3. 掃描並分類檔案
files = [f for f in os.listdir(base_dir) if f.startswith('events.out.tfevents')]
print(f"開始掃描 {len(files)} 個紀錄檔...\n")

for f in files:
    file_path = os.path.join(base_dir, f)
    # 取得檔案最後修改時間
    mtime = os.path.getmtime(file_path)
    dt_str = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')

    # 判斷邏輯
    if mtime <= mamba_end_time:
        target = 'mamba_pre'
        label = "Mamba Pretrain"
    elif mamba_end_time < mtime <= vit_end_time:
        target = 'vit_pre'
        label = "ViT Pretrain"
    else:
        target = 'vit_fine'
        label = "ViT Fine-tuning (現在)"

    # 移動檔案
    dest_path = os.path.join(paths[target], f)
    shutil.move(file_path, dest_path)
    print(f"[{dt_str}] -> 歸類至 {label}")

print("\n--- 分類完成！ ---")
print("現在你可以直接啟動 TensorBoard 了：")
print("tensorboard --logdir=./output_dir --bind_all")