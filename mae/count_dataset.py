import os
import pandas as pd
from collections import defaultdict

# 1. 設定你的資料集根目錄 (WSL 路徑)
root_dir = "/mnt/e/plant/MAE_plant/PlantCLEF2024singleplanttrainingdata_800_max_side_size/images_max_side_800"

def get_stats():
    splits = ['train', 'val', 'test']
    # 使用 dict 儲存統計數據：{ class_id: { 'train': 0, 'val': 0, 'test': 0 } }
    stats = defaultdict(lambda: {s: 0 for s in splits})
    
    print("正在掃描資料夾並統計數量，請稍候...")
    
    # 遍歷 train, val, test 資料夾
    for split in splits:
        split_path = os.path.join(root_dir, split)
        if not os.path.exists(split_path):
            print(f"找不到 {split} 資料夾，請確認路徑。")
            continue
            
        classes = os.listdir(split_path)
        for cls in classes:
            cls_path = os.path.join(split_path, cls)
            if os.path.isdir(cls_path):
                # 計算該類別下的圖片數量
                count = len([f for f in os.listdir(cls_path) if os.path.isfile(os.path.join(cls_path, f))])
                stats[cls][split] = count

    # 2. 轉換為 DataFrame 方便處理
    df = pd.DataFrame.from_dict(stats, orient='index')
    df.index.name = 'Class_ID'
    
    # 計算每一類的總數
    df['Total'] = df['train'] + df['val'] + df['test']
    
    # 計算各分區總計
    total_train = df['train'].sum()
    total_val = df['val'].sum()
    total_test = df['test'].sum()
    total_all = df['Total'].sum()

    # 3. 顯示結果
    print("\n" + "="*50)
    print("      資料集整體統計 (總量)")
    print("="*50)
    print(f"Train 總數: {total_train:,} ({total_train/total_all:.1%})")
    print(f"Val   總數: {total_val:,} ({total_val/total_all:.1%})")
    print(f"Test  總數: {total_test:,} ({total_test/total_all:.1%})")
    print(f"總計圖片數: {total_all:,}")
    print("="*50)

    print("\n[前 20 個類別的分布情況]:")
    print(df.head(20))

    # 4. 存成 CSV 檔以便查看完整名單
    csv_filename = "dataset_stats.csv"
    df.to_csv(csv_filename)
    print(f"\n✅ 完整的統計表已存為: {os.path.abspath(csv_filename)}")

if __name__ == "__main__":
    get_stats()
