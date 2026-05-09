import json
import matplotlib.pyplot as plt
import os

def load_log(path, key):
    if not os.path.exists(path):
        print(f"找不到路徑: {path}")
        return [], []
    data_dict = {}
    with open(path, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                if key in data and 'epoch' in data:
                    data_dict[data['epoch']] = data[key]
            except: continue
    epochs = sorted(data_dict.keys())
    values = [data_dict[e] for e in epochs]
    return epochs, values

# 1. 定義所有路徑
paths = {
    'mamba_pre': './output_mamba_pretrain/log.txt',
    'vit_pre': './output_vit_pretrain/log.txt',
    'mamba_fine': './output_mamba_finetune/log.txt',
    'vit_fine': './output_vit_finetune/log.txt'
}

# 2. 抓取數據
m_pre_x, m_pre_y = load_log(paths['mamba_pre'], 'train_loss')
v_pre_x, v_pre_y = load_log(paths['vit_pre'], 'train_loss')
m_fine_x, m_fine_y = load_log(paths['mamba_fine'], 'test_acc1')
v_fine_x, v_fine_y = load_log(paths['vit_fine'], 'test_acc1')

# 3. 開始畫圖 (上下兩層)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
plt.subplots_adjust(hspace=0.3)

# --- 上層：Pre-train Loss ---
ax1.plot(m_pre_x, m_pre_y, label=f'MambaVision (Final: {m_pre_y[-1]:.4f})', color='#1f77b4', linewidth=2)
ax1.plot(v_pre_x, v_pre_y, label=f'ViT-Base (Final: {v_pre_y[-1]:.4f})', color='#ff7f0e', linewidth=2)
ax1.set_title('Stage 1: MAE Pre-training (Reconstruction Loss)', fontsize=14)
ax1.set_ylabel('Loss')
ax1.legend(); ax1.grid(True, linestyle='--', alpha=0.6)

# --- 下層：Fine-tune Accuracy ---
if m_fine_y:
    ax2.plot(m_fine_x, m_fine_y, label=f'MambaVision (Max: {max(m_fine_y):.2f}%)', color='#1f77b4', marker='o', markersize=4)
if v_fine_y:
    ax2.plot(v_fine_x, v_fine_y, label=f'ViT-Base (Max: {max(v_fine_y):.2f}%)', color='#ff7f0e', marker='s', markersize=4)
ax2.set_title('Stage 2: Fine-tuning (Top-1 Accuracy)', fontsize=14)
ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy (%)')
ax2.legend(); ax2.grid(True, linestyle='--', alpha=0.6)

# 儲存結果
plt.savefig('full_experiment_report.png')
print("✅ 完整實驗報告圖已儲存為 full_experiment_report.png")