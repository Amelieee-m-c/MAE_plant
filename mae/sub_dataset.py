import os
import random
import shutil

'''
====================================================================
植物資料集 100 類【實體複製版】腳本 (WSL 專用)
====================================================================
功能說明：
1. 完好保留您原本的幾千類大資料集。
2. 聰明篩選出在 train/val/test 中「都有真實圖片」的 100 個健全類別。
3. 實體複製（Copy）這些資料夾與圖片到新目錄，讓它們在 E 槽實體存在！
====================================================================
'''

# 1. 設定 WSL 原生路徑
src_root = "/mnt/e/plant/MAE_plant/PlantCLEF2024singleplanttrainingdata_800_max_side_size/images_max_side_800"
dst_root = "/mnt/e/plant/MAE_plant/PlantCLEF_100_classes"

splits = ['train', 'val', 'test']

def main():
    if not os.path.exists(os.path.join(src_root, 'train')):
        print(f"❌ 錯誤：找不到原始訓練集路徑！")
        return

    print("🔍 步驟 1：正在掃描原始 721 資料集並篩選健全類別...")
    train_classes = set(os.listdir(os.path.join(src_root, 'train')))
    val_classes = set(os.listdir(os.path.join(src_root, 'val')))
    test_classes = set(os.listdir(os.path.join(src_root, 'test')))

    # 取交集並確保裡面真的有圖
    common_classes = train_classes.intersection(val_classes).intersection(test_classes)
    perfect_classes = []
    for cls in common_classes:
        is_perfect = True
        for split in splits:
            cls_dir = os.path.join(src_root, split, cls)
            if len([f for f in os.listdir(cls_dir) if os.path.isfile(os.path.join(cls_dir, f))]) == 0:
                is_perfect = False
                break
        if is_perfect:
            perfect_classes.append(cls)

    print(f"📈 掃描完成！共有 {len(perfect_classes)} 個健全類別。")

    # 2. 隨機抽出 100 個完美類別
    selected_classes = random.sample(perfect_classes, 100) if len(perfect_classes) >= 100 else perfect_classes
    print(f"🎲 步驟 2：已成功隨機抽選出 100 個植物類別。")

    # 3. 清理舊目錄
    if os.path.exists(dst_root):
        print("🧹 步驟 3：發現舊的 100 類資料夾，正在清理重製...")
        shutil.rmtree(dst_root)

    # 4. 開始【實體複製】
    print("🚀 步驟 4：開始實體複製資料夾與圖檔到 E 槽 (檔案較多，請稍候)...")
    for split in splits:
        print(f"   -> 正在複製 {split} 分區...")
        for cls in selected_classes:
            src_cls_dir = os.path.join(src_root, split, cls)
            dst_cls_dir = os.path.join(dst_root, split, cls)
            
            # 使用 shutil.copytree 進行硬碟實體複製
            shutil.copytree(src_cls_dir, dst_cls_dir)

    print("\n" + "="*50)
    print("🎉 ✅ 【實體版】100 類同步測試集建立完成！")
    print("="*50)
    print(f"● 新資料夾路徑: {dst_root}")
    print("現在你去 Windows 的 E 槽看，它們全都是實體檔案跟正確的 KB 數了！\n")

if __name__ == '__main__':
    main()
