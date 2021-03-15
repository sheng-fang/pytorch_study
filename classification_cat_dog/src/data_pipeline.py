import os

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch


class OxfordDogCatDS(object):
    def __init__(self, anno_df, img_dir, img_shape, transforms=None,
                 img_ext=".jpg"):
        self.anno_df = anno_df
        self.img_dir = img_dir
        self.img_ext = img_ext
        self.transforms = transforms
        self.img_shape = img_shape

    def __len__(self):
        return len(self.anno_df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir,
                                self.anno_df.iloc[idx, 0] + self.img_ext)
        label = self.anno_df.iloc[idx, 1] - 1
        img = Image.open(img_path)
        img = img.convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)
        return img, label


