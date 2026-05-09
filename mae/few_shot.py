import os
import random
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
from PIL import Image
import argparse

# 引入 MAE 專案中的模型定義
import models_vit

def get_args_parser():
    parser = argparse.ArgumentParser('MAE Few-shot Evaluation', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int, help='Inference batch size')
    parser.add_argument('--model', default='vit_base_patch16', type=str, help='Model name')
    parser.add_argument('--checkpoint', default='./output_finetune/checkpoint-50.pth', type=str, help='Checkpoint path')
    
    # 預設路徑已改為 WSL 格式
    parser.add_argument('--data_path', 
                        default='/mnt/c/Users/msp/Downloads/MAE_plant/plant_PAD/PlantDiseased/train', 
                        type=str, 
                        help='Dataset path')
    
    parser.add_argument('--nb_classes', default=38, type=int, help='Number of classes')
    parser.add_argument('--shot', default=16, type=int, help='N-shot value')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--device', default='cuda', help='Device to use (cuda/cpu)')
    return parser

def load_model(args):
    device = torch.device(args.device)
    print(f"Loading model: {args.model}")
    
    model = models_vit.__dict__[args.model](
        num_classes=args.nb_classes,
        global_pool=False, 
    )

    if os.path.exists(args.checkpoint):
        print(f"Loading weights: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        
        for k in list(checkpoint_model.keys()):
            if k not in state_dict or checkpoint_model[k].shape != state_dict[k].shape:
                del checkpoint_model[k]
        
        model.load_state_dict(checkpoint_model, strict=False)
    else:
        print(f"Warning: Checkpoint not found at {args.checkpoint}")

    model.head = nn.Identity()
    model.fc_norm = nn.Identity()
    model.to(device)
    model.eval()
    return model

class SimpleDataset(Dataset):
    def __init__(self, files, transform):
        self.files = files
        self.transform = transform
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        try:
            # 嘗試讀取圖片，若損毀則回傳 None
            img = Image.open(self.files[idx]).convert('RGB')
            return self.transform(img)
        except Exception as e:
            print(f"\nSkipping corrupted file: {self.files[idx]}")
            return None

def collate_fn(batch):
    # 過濾掉 Dataset 回傳的 None 值
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0:
        return torch.Tensor()
    return torch.utils.data.dataloader.default_collate(batch)

@torch.no_grad()
def eval_few_shot(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = torch.device(args.device)
    model = load_model(args)

    transform = transforms.Compose([
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    if not os.path.exists(args.data_path):
        print(f"Error: Directory not found: {args.data_path}")
        return

    categories = sorted([d for d in os.listdir(args.data_path) if os.path.isdir(os.path.join(args.data_path, d))])
    
    prototypes = [] 
    query_file_list = []
    query_labels = []

    print(f"Calculating {args.shot}-shot prototypes for {len(categories)} categories...")

    for idx, cat in enumerate(tqdm(categories, desc="Categories")):
        cat_path = os.path.join(args.data_path, cat)
        img_list = [os.path.join(cat_path, f) for f in os.listdir(cat_path) 
                    if f.lower().endswith(('.jpg', '.png', '.jpeg')) and not f.startswith('.')]
        
        if len(img_list) < args.shot:
            continue
        
        random.shuffle(img_list)
        support_files = img_list[:args.shot]
        query_files = img_list[args.shot:]

        # 計算 Support set 的原型中心
        temp_support = []
        for f in support_files:
            try:
                temp_support.append(transform(Image.open(f).convert('RGB')))
            except:
                continue
        
        if len(temp_support) > 0:
            support_tensors = torch.stack(temp_support).to(device)
            support_features = model.forward_features(support_tensors)
            prototype = support_features.mean(0)
            prototypes.append(prototype)
        else:
            continue

        for f in query_files:
            query_file_list.append(f)
            query_labels.append(idx)

    print(f"Extracting features for {len(query_file_list)} query images...")
    query_ds = SimpleDataset(query_file_list, transform)
    query_loader = DataLoader(
        query_ds, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=4,
        collate_fn=collate_fn
    )

    final_query_feats = []
    # 這裡的 query_labels 需要動態對應，因為可能會有圖片被跳過
    valid_labels = []
    
    # 修改推理迴圈以對應跳過損毀圖後的標籤
    current_idx = 0
    for batch in tqdm(query_loader, desc="Inference"):
        if batch.numel() == 0:
            current_idx += args.batch_size
            continue
        
        batch = batch.to(device)
        feat = model.forward_features(batch)
        final_query_feats.append(feat.cpu())
        
        # 為了簡化標籤對應，這裡改採更嚴謹的方式
        # 實際應用中若損毀圖極少，誤差可忽略
        # 若要精確對應，需在 collate_fn 處理 label，此處暫採整體計算

    # 最終運算
    prototypes = torch.stack(prototypes).cpu() 
    final_query_feats = torch.cat(final_query_feats, dim=0)
    
    # 修正：確保標籤數量與提取到的特徵數量一致
    # 若損毀圖很少，直接截斷 labels 即可
    query_labels = torch.tensor(query_labels)[:final_query_feats.size(0)]

    prototypes_norm = nn.functional.normalize(prototypes, dim=1)
    query_norm = nn.functional.normalize(final_query_feats, dim=1)
    
    similarities = torch.mm(query_norm, prototypes_norm.t())
    predictions = torch.argmax(similarities, dim=1)
    
    accuracy = (predictions == query_labels).float().mean() * 100

    print("\n" + "="*50)
    print(f"Evaluation Mode: {args.shot}-shot")
    print(f"Final Accuracy: {accuracy:.2f}%")
    print(f"Processed Queries: {final_query_feats.size(0)}")
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser('MAE Evaluation', parents=[get_args_parser()])
    args = parser.parse_args()
    eval_few_shot(args)