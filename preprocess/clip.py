import html
import os
import re
from typing import List, Union

import ftfy
import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer, BatchFeature


# taken from https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/tokenizer.py#L65C8-L65C8
def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


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


class CLIP:
    def __init__(
        self, model_path="stabilityai/japanese-stable-clip-vit-l-16", device="cuda"
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(
            device
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.processor = AutoImageProcessor.from_pretrained(model_path)
        self.model.eval()


    def get_text_features(self, texts: List[str], batch_size=200):
        text_feature_all = []
        for i in range((len(texts) - 1) // batch_size + 1):
            texts_tmp = texts[i * batch_size : (i + 1) * batch_size]
            text = tokenize(
                tokenizer=self.tokenizer,
                texts=texts_tmp,
            ).to(self.device)
            with torch.no_grad():
                text_features = self.model.get_text_features(**text)
            text_feature_all.append(text_features)
        print("text feature", torch.concat(text_feature_all).shape)
        return torch.concat(text_feature_all)

    def get_image_features(self, images: List[str], batch_size=200):
        images = [Image.open(p) for p in images]
        image_feature_all = []
        for i in range((len(images) - 1) // batch_size + 1):
            images_tmp = self.processor(
                images=images[i * batch_size : (i + 1) * batch_size],
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                image_features = self.model.get_image_features(**images_tmp)
                image_feature_all.append(image_features)
        print("image feature", torch.concat(image_feature_all).shape)
        return torch.concat(image_feature_all)

    # @profile
    def retrieve_text_from_image(self, images: List[str], texts: List[str]):
        """
        images内の各imageに対して最も近いtextをtextsから取ってくる
        """
        print(len(images), len(texts))
        image_features = self.get_image_features(images)
        text_features = self.get_text_features(texts)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        text_probs = (image_features @ text_features.T).softmax(dim=-1)
        max_inds = torch.argmax(text_probs, axis=1)
        max_inds = np.array(max_inds.cpu())
        return [texts[max_ind] for max_ind in max_inds]

    def retrieve_text_from_image_topk(self, images: List[str], texts: List[str], topk=3):
        """
        images内の各imageに対して最も近いtextをtextsから取ってくる
        """
        print(len(images), len(texts))
        image_features = self.get_image_features(images)
        text_features = self.get_text_features(texts)
        text_probs = (image_features @ text_features.T).softmax(dim=-1)
        topk = min(topk, text_probs.size(1))
        max_inds = torch.flatten(torch.topk(text_probs, dim=1, k=topk).indices)
        max_inds = np.array(max_inds.cpu())
        return [texts[max_ind] for max_ind in max_inds]

    def retrieve_image_from_text_topk(self, images: List[str], texts: List[str], topk=3):
        """
        images内の各imageに対して最も近いtextをtextsから取ってくる
        """
        print(len(images), len(texts))
        image_features = self.get_image_features(images)
        text_features = self.get_text_features(texts)
        image_probs = (text_features @ image_features.T).softmax(dim=-1)
        max_inds = torch.flatten(torch.topk(image_probs, dim=1, k=topk).indices)
        max_inds = np.array(max_inds.cpu())
        return [images[max_ind] for max_ind in max_inds]

    def retrieve_image_from_text(self, images: List[str], texts: List[str]):
        """
        images内の各imageに対して最も近いtextをtextsから取ってくる
        """
        print(len(images), len(texts))
        image_features = self.get_image_features(images)
        text_features = self.get_text_features(texts)
        image_probs = (text_features @ image_features.T).softmax(dim=-1)
        max_inds = torch.argmax(image_probs, axis=1)
        max_inds = np.array(max_inds.cpu())
        return [images[max_ind] for max_ind in max_inds]


if __name__ == "__main__":
    clip = CLIP(device="cuda")
    data_dir = "/home/yamanishi/project/trip_recommend/data/jalan_image_with_caption"
    image_features = clip.get_image_features(
        [Image.open(os.path.join(data_dir, f"旭山動物園_{i}.jpg")) for i in range(20)]
    )
    # df_review = pd.read_csv('/home/yamanishi/project/trip_recommend/data/jalan/review/review_all_period_.csv')
    df_review = pd.read_pickle(
        "/home/yamanishi/project/trip_recommend/data/jalan/review/review_all_period_.pkl"
    )
    df_tmp = df_review[df_review["spot"] == "旭山動物園"]
    reviews = []
    original_reviews = {}
    for review in df_tmp["review"]:
        review_split = review.split("。")
        for r in review_split:
            original_reviews[r] = review
        reviews += review_split

    text_features = clip.get_text_features(reviews)
    text_probs = (image_features @ text_features.T).softmax(dim=-1)
    print(np.array(text_probs.cpu()))
    max_inds = np.argmax(np.array(text_probs.cpu()), axis=1)
    for i, max_ind in enumerate(max_inds):
        print(i, reviews[max_ind], original_reviews[reviews[max_ind]])
