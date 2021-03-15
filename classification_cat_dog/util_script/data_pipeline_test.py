import os
import pathlib
import itertools

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
import cv2

from classification_cat_dog.src import data_pipeline
from classification_cat_dog.src import models


BATCH_SIZE = 32
NB_EPO = 3
LR = 0.01


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy()#.transpose((1, 2, 0))
    # mean = np.array([0.485, 0.456, 0.406])
    # std = np.array([0.229, 0.224, 0.225])
    # inp = std * inp + mean
    # inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


img_shape = (224, 224)
dat_rel_dir = "data/kaggle/oxfordCatDog"
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

transforms = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(),
     torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_ds = data_pipeline.OxfordDogCatDS(
    anno_train_flt, img_dir, img_shape, transforms=transforms)

test_ds = data_pipeline.OxfordDogCatDS(
    anno_test_flt, img_dir, img_shape, transforms=transforms
)

train_dsld = torch.utils.data.DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

test_dsld = torch.utils.data.DataLoader(
    test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
)

# for imgs, labels in iter(train_dsld):
#     tmp = imgs.numpy()
#     for idx in range(BATCH_SIZE):
#         img_toshow = tmp[idx, ]
#         cv2.imshow("img", img_toshow)
#         cv2.waitKey(0)


resnet18 = torchvision.models.resnet18(pretrained=True)

fc_head = models.FCHead([512, 100, 37])
pass_head = models.PassHead()

resnet18.fc = pass_head
# print(resnet18)
resnet18.train()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
resnet18.to(device)
fc_head.to(device)

criterion = nn.CrossEntropyLoss()

last_param = None
last_param_fc = None

for param in fc_head.parameters():
    last_param_fc = torch.sum(param.data).cpu().numpy()
    break

for param in resnet18.layer1.parameters():
    last_param = torch.sum(param.data).cpu().numpy()
    break

optim = torch.optim.SGD(
    itertools.chain(resnet18.parameters(), fc_head.parameters()),
    lr=LR, momentum=0.9)


for epo in range(NB_EPO):
    epo_loss = []
    for data, label in iter(train_dsld):
        data = data.to(device)
        label = label.to(device)

        optim.zero_grad()
        with torch.no_grad():
            emb = resnet18(data)
        pred = fc_head(emb)

        loss = criterion(pred, label)

        loss.backward()
        optim.step()

        epo_loss.append(loss.item())

        # print("Epoch: {}, loss: {}".format(epo, np.mean(epo_loss)))
    for param in fc_head.parameters():
        curr_param = torch.sum(param.data).cpu().numpy()
        if curr_param == last_param_fc:
            print("Resnet fc Unchanged.")
        else:
            last_param_fc = curr_param
            print("Resnet fc Updated")
        break

    for param in resnet18.layer1.parameters():
        curr_param = torch.sum(param.data).cpu().numpy()
        if curr_param == last_param:
            print("Resnet Unchanged.")
        else:
            last_param = curr_param
            print("Resnet Updated")
        break

    print("Epoch {} loss: {}".format(epo, np.mean(epo_loss)))



