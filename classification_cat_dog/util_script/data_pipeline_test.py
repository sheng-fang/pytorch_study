import os
import pathlib

import numpy as np
import pandas as pd
import torch
import torchvision
import matplotlib.pyplot as plt

from classification_cat_dog.src import data_pipeline


img_shape = (224, 224)
dat_rel_dir = "Data/kaggle/oxfordCatDog"
data_dir = pathlib.Path(os.path.join(os.path.expanduser("~"), dat_rel_dir))
img_dir = data_dir / "images"
anno_train_path = data_dir / "annotations" / "trainval.txt"
anno_test_path = data_dir / "annotations" / "test.txt"
anno_train = np.loadtxt(anno_train_path, dtype=object)
anno_test = np.loadtxt(anno_test_path, dtype=object)
anno_train_flt = pd.DataFrame([[x[0], int(x[1])] for x in anno_train],
                              columns=["img_name", "label"])
anno_test_flt = pd.DataFrame([[x[0], int(x[1])] for x in anno_test],
                             columns=["img_name", "label"])

train_ds = data_pipeline.OxfordDogCatDS(anno_train_flt, img_dir, img_shape)
# train_ds = iter(train_ds)
train_dsld = torch.utils.data.DataLoader(
    train_ds, batch_size=10, shuffle=True, num_workers=4)

transforms = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor,
     torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


