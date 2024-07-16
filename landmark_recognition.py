import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torchvision.models as models
import time
import copy
import pickle
from tqdm import tqdm
import argparse
import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from PIL import ImageFile
import random
ImageFile.LOAD_TRUNCATED_IMAGES = True

def set_seed(seed):
    # Python標準の乱数生成器
    random.seed(seed)
    
    # NumPyの乱数生成器
    np.random.seed(seed)
    
    # PyTorchの乱数生成器
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 全てのGPUに対して設定

    # TensorFlowの乱数生成器
    #tf.random.set_seed(seed)
    
    # 一部のライブラリでは、再現性を完全に保証するための追加の設定が必要です（例：PyTorch）
    # デフォルトでは、PyTorchは非決定的なアルゴリズムを使用することがあります
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
class LandmarkDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform
        self.image_dir = "/home/yamanishi/project/trip_recommend/data/jalan_image_with_caption"

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        img_path = row['image_path']
        img_path = os.path.join(self.image_dir, img_path)
        image = Image.open(img_path).convert("RGB")
        label = row['label']

        if self.transform:
            image = self.transform(image)

        return image, label


def train(args):
    set_seed(42)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # データセットの作成
    train_df = pd.read_csv('./data/landmark_recognition_train.csv')
    test_df = pd.read_csv('./data/landmark_recognition_test.csv')
    eval_df = pd.read_csv('./data/landmark_recognition_eval.csv')
    llavatour_df = pd.read_csv('./result/spot_name/llavatour.csv')
    test_df = test_df[test_df['image_path'].isin(llavatour_df['image_path'].values)]
    print('len test df', len(test_df))
    #test_df = test_df.sample(1000)
    train_df, val_df = train_test_split(train_df, test_size=0.2)

    train_dataset = LandmarkDataset(dataframe=train_df, transform=transform)
    val_dataset = LandmarkDataset(dataframe=val_df, transform=transform)
    test_dataset = LandmarkDataset(dataframe=test_df, transform=transform)
    eval_dataset = LandmarkDataset(dataframe=eval_df, transform=transform)
    # データローダーの作成
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    eval_loader = DataLoader(eval_dataset, batch_size=64, shuffle=False)
    
    if args.model_name == 'resnet18':
        model = models.resnet18(pretrained=True)
    elif args.model_name == 'resnet50':
        model = models.resnet50(pretrained=True)
    elif args.model_name == 'resnet101':
        model = models.resnet101(pretrained=True)
    elif args.model_name == 'vit_b_16':
        model = models.vit_b_16(pretrained=True)
    elif args.model_name == 'vit_b_32':
        model = models.vit_b_32(pretrained=True)
    elif args.model_name == 'vit_l_16':
        model = models.vit_l_16(pretrained=True)
    elif args.model_name == 'vit_l_32':
        model = models.vit_l_32(pretrained=True)
        
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, max(train_df['label'].max(), test_df['label'].max())+1)
    # デバイスの設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    if args.checkpoint is not None:
        model.load_state_dict(torch.load(args.checkpoint))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    model_save_dir = f'checkpoint/landmark_recognition/{args.exp_id}'
    os.makedirs(model_save_dir, exist_ok=True)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    num_epochs = 25
    if args.eval_only:
        accuracy, precision, recall, f1, all_labels, all_predictions= evaluate_model(model, eval_loader, 'cuda')
        with open(f'result/spot_name/resnet_{args.exp_id}.pkl', 'wb') as f:
            pickle.dump({'gt': all_labels, 'pred': all_predictions}, f)
        return
        
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        model.train()
        running_loss = 0.0
        running_corrects = 0
        running_loss_tmp = 0
        running_corrects_tmp = 0
        data_size_tmp = 0
        data_size = 0

        # データローダーからデータをロード
        for i,(inputs, labels) in tqdm(enumerate(train_loader)):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # 勾配のリセット
            optimizer.zero_grad()

            # 順伝播
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            # 逆伝播と最適化
            loss.backward()
            optimizer.step()

            # 損失と正解数の合計を計算
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            running_loss_tmp += loss.item() * inputs.size(0)
            running_corrects_tmp += torch.sum(preds == labels.data)
            data_size_tmp += inputs.size(0)
            data_size += inputs.size(0)
            
            if i%50==0:
                running_loss_tmp_mean = running_loss/data_size_tmp
                running_correct_tmp_mean = running_corrects_tmp/data_size_tmp
                print(f'Loss: {running_loss_tmp_mean:.4f}, Accuracy:{running_correct_tmp_mean:.4f}')
                running_loss_tmp = 0
                running_corrects_tmp = 0
                data_size_tmp = 0
            #if i==10:break
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)

        print(f'Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        # ベストモデルのディープコピー
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
        accuracy, precision, recall, f1, _, _= evaluate_model(model, val_loader, 'cuda')
        print(f'Accuracy: {accuracy:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'F1: {f1:.4f}')
        metric = evaluate_model(model, eval_loader, 'cuda')
        print('eval metric', metric[:4])
        model_save_path = os.path.join(model_save_dir, f'epoch_{epoch+1}.pth')
        torch.save(model.state_dict(), model_save_path)
        print()

    print(f'Best Acc: {best_acc:4f}')

    # ベストモデルの重みをロード
    model.load_state_dict(best_model_wts)
    
    return model

def evaluate_model(model, test_loader, device):
    model.eval()
    all_labels = []
    all_predictions = []
    print(len(test_loader))
    with torch.no_grad():
        for i,(images, labels) in tqdm(enumerate(test_loader)):
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(preds.cpu().numpy())
            #if i==10:break
    
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    #print('all_predictions', all_predictions)
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='weighted')
    recall = recall_score(all_labels, all_predictions, average='weighted')
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    confusion = confusion_matrix(all_labels, all_predictions)
    
    return accuracy, precision, recall, f1, all_labels, all_predictions

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_id', type=str, default='image_only')
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--model_name', type=str, default='resnet18')
    args = parser.parse_args()
    train(args)