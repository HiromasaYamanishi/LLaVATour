import pandas as pd
import numpy as np
import torch
#mport clip
from PIL import Image
from torchvision import transforms
from sklearn.model_selection import train_test_split
from typing import List, Union
import os
import torch.nn as nn
import ftfy
import html
import random
import re
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer, BatchFeature
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import argparse
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error
def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()

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
    
def whitespace_clean(text):
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text

def tokenize(
    tokenizer,
    texts: Union[str, List[str]],
    max_seq_len: int = 77,
):
    """
    This is a function that have the original clip's code has.
    https://github.com/openai/CLIP/blob/main/clip/clip.py#L195
    """
    if isinstance(texts, str):
        texts = [texts]
    texts = [whitespace_clean(basic_clean(text)) for text in texts]

    inputs = tokenizer(
        texts,
        max_length=max_seq_len - 1,
        padding="max_length",
        truncation=True,
        add_special_tokens=False,
    )
    # add bos token at first place
    input_ids = [[tokenizer.bos_token_id] + ids for ids in inputs["input_ids"]]
    attention_mask = [[1] + am for am in inputs["attention_mask"]]
    position_ids = [list(range(0, len(input_ids[0])))] * len(texts)

    return BatchFeature(
        {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "position_ids": torch.tensor(position_ids, dtype=torch.long),
        }
    )


class ImageCaptionDataset(Dataset):
    def __init__(self, dataframe, clip_model, args, transform=None):
        self.dataframe = dataframe
        self.transform = transform
        self.image_dir = "/home/yamanishi/project/trip_recommend/data/jalan_image_with_caption"
        self.use_text = args.use_text
        self.use_spot_name = args.use_spot_name
        self.processor = AutoImageProcessor.from_pretrained(clip_model)
        self.device = 'cuda'

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image = Image.open(os.path.join(self.image_dir, row['id'])).convert('RGB')
        image = self.processor(
            images=[image],
            return_tensors="pt",
        ).to(self.device)
        # if self.transform:
        #     image = self.transform(image)
        if self.use_text:
            caption = row['text']
        if self.use_spot_name:
            spot_name = row['spot_name']
        likes = torch.tensor(row['tag'], dtype=torch.float32)
        image = image['pixel_values'].squeeze(0)
        if self.use_text and self.use_spot_name:
            return image, caption, spot_name, likes
        elif self.use_text:
            return image, caption, likes
        else:
            return image, likes
    
class CLIPRegression(nn.Module):
    def __init__(self, args):
        super(CLIPRegression, self).__init__()
        device = 'cuda'

        self.device = device
        if 'stabilityai' in args.clip_name:
            self.model = AutoModel.from_pretrained(args.clip_name, trust_remote_code=True).to(
                device
            )
            self.tokenizer = AutoTokenizer.from_pretrained(args.clip_name)
            self.processor = AutoImageProcessor.from_pretrained(args.clip_name)
            out_dim = 768
            if args.use_text:
                out_dim += 768
            if args.use_spot_name:
                self.spot_embedding = torch.nn.Embedding(32000, 384)
                self.lin1 = torch.nn.Linear(384, 384)
                # self.mlp = torch.nn.ModuleList([torch.nn.Linear(384, 384),
                #                             torch.nn.ReLU(),
                #                             torch.nn.Linear(384, 384)])
                out_dim += 384

        self.fc_regression = nn.Linear(out_dim, 1)
        self.fc_classification = nn.Linear(out_dim, 1)
        self.use_text = args.use_text
        self.use_spot_name = args.use_spot_name
    
    def forward(self, images, texts=None, spot_labels=None):
        images = {'pixel_values': images}
        image_features = self.model.get_image_features(**images)
        
        if self.use_text:
            text = tokenize(
                tokenizer=self.tokenizer,
                texts=texts,
            ).to(self.device)
            text_features = self.model.get_text_features(**text)
            
            image_features = torch.cat([image_features, text_features], dim=1)
            
        if self.use_spot_name:
            spot_emb = self.spot_embedding(spot_labels)
            spot_emb = self.lin1(spot_emb)
            image_features = torch.cat([image_features, spot_emb], dim=1)
            
        regression_output = self.fc_regression(image_features)
        classification_output = torch.sigmoid(self.fc_classification(image_features))
        return regression_output, classification_output
    
def train(args):
    set_seed(42)
    use_text = args.use_text
    use_spot_name = args.use_spot_name
    batch_size = args.batch_size
    clip_name = args.clip_name
    clip_model = CLIPRegression(args=args)
    clip_model.to('cuda')
    if args.checkpoint is not None:
        clip_model.load_state_dict(torch.load(args.checkpoint))
    train_df = pd.read_csv('./data/df_meta_train.csv')
    test_df = pd.read_csv('./data/df_meta_test.csv')
    le = LabelEncoder()
    le.fit(pd.concat([train_df, test_df])['spot_name'])
    train_df['spot_name'] = le.transform(train_df['spot_name'])
    test_df['spot_name'] = le.transform(test_df['spot_name'])
    train_df_under10 = train_df[train_df['tag']<10]
    train_df_over10 = train_df[train_df['tag']>=10]
    train_df = pd.concat([train_df_over10, train_df_under10[:len(train_df_over10)*2]])
    train_df, val_df = train_test_split(train_df, test_size=0.2)
    train_df['text'] = train_df['text'].fillna('キャプションなし')
    val_df['text'] = val_df['text'].fillna('キャプションなし')
    test_df['text'] = test_df['text'].fillna('キャプションなし')

    model_save_dir = f'checkpoint/tag_count/{args.exp_id}'
    os.makedirs(model_save_dir, exist_ok=True)
    # データローダーの作成
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
    ])

    train_dataset = ImageCaptionDataset(train_df, clip_model=clip_name,transform=preprocess, args=args)
    val_dataset = ImageCaptionDataset(val_df,clip_model=clip_name, transform=preprocess, args=args)
    test_dataset = ImageCaptionDataset(test_df,clip_model=clip_name, transform=preprocess, args=args)
    print(len(train_dataset), len(val_dataset), len(test_dataset))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    regression_criterion = nn.MSELoss()
    classification_criterion = nn.BCELoss()
    optimizer = optim.Adam(clip_model.parameters(), lr=1e-4)

    # 学習ループ
    num_epochs = 10
    if args.eval_only:
        test_df_under10 = test_df[test_df['tag']<10]
        test_df_over10 = test_df[test_df['tag']>=10]
        test_df = pd.concat([test_df_over10, test_df_under10.sample(len(test_df_over10)*2)])
        eval_df = test_df.sample(3000)
        eval_df.to_csv('./data/df_meta_eval.csv')
        eval_dataset = ImageCaptionDataset(eval_df,clip_model=clip_name, transform=preprocess, args=args)
        eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
        #clip_model.load_state_dict(torch.load('./checkpoint/image_text/epoch_9.pth'))
        accuracy, precision, recall, f1, roc_auc, corr, mae, rmse = evaluate_model(clip_model, test_loader, args,'cuda')
        print(f'Accuracy: {accuracy:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'F1: {f1:.4f}')
        print(f'ROC_AUC: {roc_auc:.4f}')
        print(f'CORR: {corr:.4f}')
        print(f'MAE:{mae:4f}')
        print(f'RMSE:{rmse:.4f}')
        return
    
    for epoch in range(num_epochs):
        clip_model.train()
        running_loss, running_reg_loss, running_cls_loss = 0.0, 0.0, 0.0
        data_size, data_size_tmp = 0, 0
        loss_tmp, reg_loss_tmp, cls_loss_tmp = 0, 0, 0
        #print(len(train_loader))
        for i,batch in tqdm(enumerate(train_loader)):
            if use_text and use_spot_name:
                images, texts, spot_names, likes = batch
                spot_names = spot_names.to(clip_model.device)
            elif use_text:
                images, texts, likes = batch
            else:
                images, likes = batch
                
            images, likes = images.to(clip_model.device), likes.to(clip_model.device)

            # いいね数が10以上かどうかのラベルを作成
            labels = (likes >= 10).float()

            # 勾配をリセット
            optimizer.zero_grad()

            # 順伝播
            if use_text and use_spot_name:
                regression_outputs, classification_outputs = clip_model(images, texts, spot_names)
            elif use_text:
                regression_outputs, classification_outputs = clip_model(images, texts)
            else:
                regression_outputs, classification_outputs = clip_model(images)
            regression_outputs = regression_outputs.squeeze()
            classification_outputs = classification_outputs.squeeze()

            # 損失の計算
            likes = torch.log10(likes+1)
            regression_loss = regression_criterion(regression_outputs, likes)
            classification_loss = classification_criterion(classification_outputs, labels)
            loss = regression_loss + classification_loss

            # 逆伝播
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            running_cls_loss += classification_loss.item() * images.size(0)
            running_reg_loss += regression_loss.item() * images.size(0)
            loss_tmp+=loss.item() * images.size(0)
            reg_loss_tmp+=regression_loss.item() * images.size(0)
            cls_loss_tmp+=classification_loss.item() * images.size(0)
            
            data_size += images.size(0)
            data_size_tmp += images.size(0)
            if i%50==0:
                loss_tmp_mean = loss_tmp/data_size_tmp
                cls_loss_tmp_mean = cls_loss_tmp/data_size_tmp
                reg_loss_tmp_mean = reg_loss_tmp/data_size_tmp
                print(f'Loss: {loss_tmp_mean:.4f}, CLS: {cls_loss_tmp_mean:.4f}, REG: {reg_loss_tmp_mean:.4f}')
                loss_tmp, cls_loss_tmp, reg_loss_tmp = 0, 0, 0
                data_size_tmp = 0
            #if i==10:break
        epoch_loss = running_loss / data_size
        epoch_reg_loss = running_reg_loss / data_size
        epoch_cls_loss = running_cls_loss / data_size
        accuracy, precision, recall, f1, roc_auc, corr, mae, rmse= evaluate_model(clip_model, val_loader, args,'cuda')
        print(f'Accuracy: {accuracy:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'F1: {f1:.4f}')
        print(f'ROC_AUC: {roc_auc:.4f}')
        print(f'CORR: {corr:.4f}')
        print(f'MAE:{mae:4f}')
        print(f'RMSE:{rmse:.4f}')
        model_save_path = os.path.join(model_save_dir, f'epoch_{epoch+1}.pth')
        torch.save(clip_model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, CLS: {epoch_cls_loss:.4f}, REG: {epoch_reg_loss:.4f}")

    clip_model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for images, captions, likes in test_loader:
            images, likes = images.to(clip_model.device), likes.to(clip_model.device)
            labels = (likes >= 10).float()
            regression_outputs, classification_outputs = clip_model(images, captions)
            regression_outputs = regression_outputs.squeeze()
            classification_outputs = classification_outputs.squeeze()
            regression_loss = regression_criterion(regression_outputs, likes)
            classification_loss = classification_criterion(classification_outputs, labels)
            loss = regression_loss + classification_loss
            test_loss += loss.item() * images.size(0)

    test_loss = test_loss / len(test_loader.dataset)
    print(f"Test Loss: {test_loss:.4f}")


    def predict_likes(image_path, caption, model, clip_model):
        image = preprocess(Image.open(image_path).convert('RGB')).unsqueeze(0).to(clip_model.device)
        with torch.no_grad():
            predicted_likes = model(image, [caption]).item()
        return predicted_likes

    # 例としての推論
    image_path = 'path_to_image.jpg'
    caption = 'キャプションの例'
    predicted_likes = predict_likes(image_path, caption, clip_model, clip_model)
    print(f"Predicted Likes: {predicted_likes}")
    
# テストデータに対する予測と評価指標の計算
def evaluate_model(model, test_loader,args, device):
    model.eval()
    all_labels = []
    all_likes = []
    all_predictions = []
    all_pred_likes = []
    
    with torch.no_grad():
        for i,batch in tqdm(enumerate(test_loader)):
            if args.use_text and args.use_spot_name:
                images, captions, spot_names, likes = batch
                spot_names = spot_names.to(model.device)
            elif args.use_text:
                images, captions,likes = batch
            else:
                images, likes = batch
            images, likes = images.to(device), likes.to(device)
            labels = (likes >= 10).float()
            all_likes.append(torch.log10(likes+1).cpu().numpy())
            if args.use_text and args.use_spot_name:
                regression_outputs, classification_outputs = model(images.to(device), captions, spot_names.to(device))
            elif args.use_text:
                regression_outputs, classification_outputs = model(images.to(device), captions)
            else:
                regression_outputs, classification_outputs = model(images.to(device))

            classification_outputs = classification_outputs.squeeze()
            
            all_labels.append(labels.cpu().numpy())
            all_predictions.append(classification_outputs.cpu().numpy())
            all_pred_likes.append(regression_outputs.squeeze().cpu().numpy())
            #if i==10:break
    
    all_labels = np.concatenate(all_labels)
    all_likes = np.concatenate(all_likes)
    all_predictions = np.concatenate(all_predictions)
    all_pred_likes = np.concatenate(all_pred_likes)
    print(all_labels.shape, all_likes.shape, all_predictions.shape, all_pred_likes.shape)
    all_predictions_binary = (all_predictions > 0.5).astype(int)  # 二値分類のための閾値処理

    accuracy = accuracy_score(all_labels, all_predictions_binary)
    precision = precision_score(all_labels, all_predictions_binary)
    recall = recall_score(all_labels, all_predictions_binary)
    f1 = f1_score(all_labels, all_predictions_binary)
    roc_auc = roc_auc_score(all_labels, all_predictions)
    confusion = confusion_matrix(all_labels, all_predictions_binary)
    corr = np.corrcoef(all_likes, all_pred_likes)[0][1]
    mae = mean_absolute_error(all_pred_likes, all_likes)
    rmse = mean_squared_error(all_pred_likes, all_likes, squared=True)
    print(all_pred_likes, all_likes)
    return accuracy, precision, recall, f1, roc_auc, corr, mae, rmse
    
# データの読み込みと分割
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_text', action='store_true')
    parser.add_argument('--use_spot_name', action='store_true')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--clip_name', type=str, default="stabilityai/japanese-stable-clip-vit-l-16")
    parser.add_argument('--exp_id', type=str, default='image_only')
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--checkpoint', type=str, default=None)
    args = parser.parse_args()
    train(args)
