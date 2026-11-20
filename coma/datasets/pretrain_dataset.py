# from MGCA
import os
import pickle
import re

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from nltk.tokenize import RegexpTokenizer
from coma.constants import *
from coma.datasets.utils import get_imgs
from tqdm import tqdm
from transformers import BertTokenizer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class MultimodalPretrainingDataset(data.Dataset):
    def __init__(self, split="train", transform=None, data_pct=1.0, rewrite=False, concept_decompose=False,
                 imsize=256, max_words=30, sent_num=3):
        super().__init__()
        if not os.path.exists(MIMIC_CXR_DATA_DIR):
            raise RuntimeError(f"{MIMIC_CXR_DATA_DIR} does not exist!")

        self.transform = transform
        self.imsize = imsize
        self.rewrite = rewrite
        self.concept_decompose = concept_decompose
        task=''
        if rewrite:
            self.df = pd.read_csv(MIMIC_CXR_MASTER_CCD_CSV)
            task='_rewrite'
        else:
            self.df = pd.read_csv(MIMIC_CXR_MASTER_CHECK_CSV)
        self.df = self.df[self.df["ViewPosition"].isin(["PA", "AP"])]
        self.df[MIMIC_CXR_PATH_COL] = self.df[MIMIC_CXR_PATH_COL].apply(
            lambda x: os.path.join(MIMIC_CXR_DATA_DIR, "/".join(x.split("/")[1:])))

        # load studies and study to text mapping
        self.filenames, self.path2sent = self.load_text_data(split,task=task)
        
        self.df = self.df[self.df[MIMIC_CXR_SPLIT_COL] == split]
        if data_pct != 1.0 and split == "train":
            self.df = self.df.sample(frac=data_pct, random_state=42)
        self.df.reset_index(drop=True, inplace=True)
        
        self.tokenizer = BertTokenizer.from_pretrained(
            # "emilyalsentzer/Bio_ClinicalBERT")
            os.path.join(BASE_DIR, '../../model_files/Bio_ClinicalBERT'))
        self.max_words = max_words

    def load_text_data(self, split, task=''):
        # get study to captions mapping
        filepath = os.path.join(
            BASE_DIR, f"../../data/captions{task}.pickle")
        if not os.path.isfile(filepath):
            print(
                f"Caption file {filepath} does not exit. Creating captions...")
            path2sent = self.create_path_2_sent_mapping()
            with open(filepath, "wb") as f:
                pickle.dump(path2sent, f, protocol=2)
                print("Save to: ", filepath)
        else:
            with open(filepath, "rb") as f:
                path2sent = pickle.load(f)

        # filter studies to use for current split
        filenames = []
        for row in self.df.itertuples():
            cur_split = getattr(row, MIMIC_CXR_SPLIT_COL)
            path = getattr(row, MIMIC_CXR_PATH_COL)
            if cur_split == split and path in path2sent:
                filenames.append(path)

        return filenames, path2sent

    def create_path_2_sent_mapping(self):
        sent_lens, num_sents = [], []
        path2sent = {}
        # iterrows is not faster than itertuples ...  but it is ok
        for _, row in tqdm(self.df.iterrows(), total=self.df.shape[0]):
            # pick impression, findings, last_paragraph
            captions = ""
            if self.rewrite:
                captions += row["text"]
            else:
                captions += row["impression"]
                captions += " "
                captions += row["findings"]

            # use space instead of newline
            captions = captions.replace("\n", " ")

            # split sentences
            splitter = re.compile("[0-9]+\.")
            captions = splitter.split(captions)
            captions = [point.split(".") for point in captions]
            captions = [sent for point in captions for sent in point]

            cnt = 0
            study_sent = []
            # create tokens from captions
            for cap in captions:
                if len(cap) == 0:
                    continue

                cap = cap.replace("\ufffd\ufffd", " ")
                # picks out sequences of alphanumeric characters as tokens
                # and drops everything else
                tokenizer = RegexpTokenizer(r"\w+")
                tokens = tokenizer.tokenize(cap.lower())
                # TODO: < 3 has instances of ['no', 'pneumothorax'], ['clear', 'lung']
                if len(tokens) <= 1:
                    continue

                # filter tokens for current sentence
                included_tokens = []
                for t in tokens:
                    t = t.encode("ascii", "ignore").decode("ascii")
                    if len(t) > 0:
                        included_tokens.append(t)

                if len(included_tokens) > 0:
                    study_sent.append(" ".join(included_tokens))

                cnt += len(included_tokens)

            if cnt >= 3:
                sent_lens.append(cnt)
                num_sents.append(len(study_sent))
                path2sent[row[MIMIC_CXR_PATH_COL]] = study_sent

        # get report word/setence statistics
        sent_lens = np.array(sent_lens)
        num_sents = np.array(num_sents)

        print(
            f"sent lens: {sent_lens.min()},{sent_lens.mean()},{sent_lens.max()} [{np.percentile(sent_lens, 5)}, {np.percentile(sent_lens, 95)}]"
        )
        print(
            f"num sents: {num_sents.min()},{num_sents.mean()},{num_sents.max()} [{np.percentile(num_sents, 5)}, {np.percentile(num_sents, 95)}]"
        )

        return path2sent

    def __len__(self):
        return len(self.filenames)

    def get_caption(self, path):
        series_sents = self.path2sent[path]

        if len(series_sents) == 0:
            raise Exception("no sentence for path")

        # separate different sentences
        series_sents = list(filter(lambda x: x != "", series_sents))

        if self.concept_decompose:
            tokens = self.tokenizer(
                series_sents,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=self.max_words,
            )
            x_len = [len([t for t in senttokens if t != 0]) for senttokens in tokens["input_ids"]]
        else:
            sent = " ".join(series_sents)
            tokens = self.tokenizer(
                sent,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=self.max_words,
            )
            x_len = [len([t for t in tokens["input_ids"][0] if t != 0])]

        return tokens, x_len

    def __getitem__(self, index):
        key = self.filenames[index]
        caps, cap_len = self.get_caption(key)
        imgs = get_imgs(key, self.imsize, self.transform, multiscale=False)
        return imgs, caps, cap_len, key


def multimodal_collate_fn(batch):
    """sort sequence"""
    imgs, cap_len, ids, reportindex, tokens, attention = [], [], [], [], [], []
    path = []
    for i,b in enumerate(batch):
        img, cap, cap_l, p = b
        imgs.append(img)
        cap_len.append(cap_l)
        ids.append(cap["input_ids"])
        reportindex.append([i for _ in range(len(cap_l))])
        tokens.append(torch.zeros_like(cap["token_type_ids"]))
        attention.append(cap["attention_mask"])
        path.append(p)

    # stack
    imgs = torch.stack(imgs)
    ids = torch.vstack(ids).squeeze()
    tokens = torch.vstack(tokens).squeeze()
    attention = torch.vstack(attention).squeeze()

    path = np.array(path)

    return_dict = {
        "caption_ids": ids,
        "reportindex": [i for sublist in reportindex for i in sublist],
        "token_type_ids": tokens,
        "attention_mask": attention,
        "imgs": imgs,
        "cap_lens": [i for sublist in cap_len for i in sublist],
        "path": path
    }
    return return_dict


if __name__ == "__main__":
    from .transforms import DataTransforms
    transform = DataTransforms(is_train=True)
    dataset = MultimodalPretrainingDataset(split="train", transform=transform)
    data = dataset[0]
    print(data)
