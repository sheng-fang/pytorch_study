import os
import pathlib
import itertools
from datetime import datetime
import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from tqdm import tqdm

from classification_cat_dog.src import data_pipeline
from classification_cat_dog.src import models
from classification_cat_dog.src import util

BATCH_SIZE = 64
NB_EPO = 9
LR = 0.05
FC_TRAINED = "fc_trained.pt"


def eval_model(resnet, test_dsld):
    preds = []
    labels = []
    test_loss = []
    resnet.eval()
    for data, label in tqdm(iter(test_dsld)):
        data = data.to(device)
        label = label.to(device)

        with torch.no_grad():
            pred = resnet(data)

        loss = criterion(pred, label)

        test_loss.append(loss.item())
        preds.append(np.argmax(pred.cpu().numpy(), axis=1))
        labels.append(label.cpu().numpy())
    preds = np.concatenate(preds, axis=0).reshape(-1, 1)
    labels = np.concatenate(labels, axis=0).reshape(-1, 1)
    acc = np.sum(preds == labels) / len(labels)
    return np.sum(test_loss), acc


now = datetime.now()
data_time = now.strftime("%Y%m%d_%H%M%S")
logger = util.get_logger(data_time + ".log", logging.INFO)

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
    [torchvision.transforms.Resize(300),
     torchvision.transforms.RandomCrop(224),
     torchvision.transforms.RandomHorizontalFlip(),
     # torchvision.transforms.RandomVerticalFlip(),
     torchvision.transforms.ToTensor(),
     torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_ds = data_pipeline.OxfordDogCatDS(
    anno_train_flt, img_dir, img_shape, transforms=transforms)

test_ds = data_pipeline.OxfordDogCatDS(
    anno_test_flt, img_dir, img_shape, transforms=transforms
)

train_size = len(train_ds)
test_size = len(test_ds)

train_dsld = torch.utils.data.DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

test_dsld = torch.utils.data.DataLoader(
    test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
)

resnet = torchvision.models.resnet50(pretrained=True)
# print(resnet)
fc_head = models.FCHead(
    [resnet.fc.in_features, resnet.fc.in_features // 2, 37]
)

resnet.fc = fc_head
# print(resnet18)
resnet.train()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
resnet.to(device)

criterion = nn.CrossEntropyLoss()


model = resnet
lr_init = 1e-5
lr_end = 10
lr_nb = 20
dsloader = train_dsld
optim = torch.optim.SGD(resnet.fc.parameters(), lr=LR, momentum=0.9)

resnet, loss_list, lr_list = util.find_lr(
    resnet, train_dsld, optim, criterion, device, lr_init=1e-2, lr_end=5,
    lr_nb=20
)

plt.figure()
plt.plot(lr_list, loss_list)
plt.xlabel("LR")
plt.ylabel("Training loss")
plt.show()

if os.path.exists(FC_TRAINED):
    resnet.load_state_dict(torch.load(FC_TRAINED))
else:
    optim = torch.optim.SGD(resnet.fc.parameters(), lr=LR, momentum=0.9)
    logger.info(
        "Set --> LR: {}, Batchsize: {}, Eporch: {}".format(
            LR, BATCH_SIZE, NB_EPO)
    )
    for epo in range(NB_EPO):
        epo_loss = []
        preds = []
        labels = []
        if epo == 2 * NB_EPO // 3:
            for param in optim.param_groups:
                param["lr"] = LR / 10.
            logger.info("Set LR: {} -> {}".format(LR, LR / 10.))
        for data, label in tqdm(iter(train_dsld)):
            # print(data.size())
            data = data.to(device)
            label = label.to(device)

            optim.zero_grad()
            pred = resnet(data)

            loss = criterion(pred, label)

            loss.backward()
            optim.step()

            epo_loss.append(loss.item())
            # preds.append(np.argmax(pred.cpu().numpy(), axis=1))
            # labels.append(label.cpu().numpy())
        #
        # preds = np.concatenate(preds, axis=0).reshape(-1, 1)
        # labels = np.concatenate(labels, axis=0).reshape(-1, 1)
        # acc = np.sum(preds == labels) / len(labels)
        # print("Epoch: {}, loss: {}".format(epo, np.mean(epo_loss)))

        logger.info(
            "Epoch {} Training loss: {}".format(
                epo, np.sum(epo_loss) / train_size)
        )
        test_loss, acc = eval_model(resnet, test_dsld)
        logger.info(
            "Epoch {} Test loss: {}, ACC: {}".format(
                epo, test_loss / test_size, acc))

    torch.save(resnet.state_dict(), "fc_trained.pt")


LR = 0.00001
NB_EPO = 9
BATCH_SIZE = 64
optim = torch.optim.SGD(resnet.parameters(), lr=LR, momentum=0.9)

logger.info(
    "Set --> LR: {}, Batchsize: {}, Eporch: {}".format(LR, BATCH_SIZE, NB_EPO))


for epo in range(NB_EPO):
    resnet.train()
    epo_loss = []
    preds = []
    labels = []
    if epo == 2 * NB_EPO // 3:
        for param in optim.param_groups:
            param["lr"] = LR / 10.
        logger.info("Set LR: {} -> {}".format(LR, LR / 10.))

    for data, label in tqdm(iter(train_dsld)):
        data = data.to(device)
        label = label.to(device)

        optim.zero_grad()
        pred = resnet(data)

        loss = criterion(pred, label)

        loss.backward()
        optim.step()

        epo_loss.append(loss.item())
        # preds.append(np.argmax(pred.cpu().numpy(), axis=1))
        # labels.append(label.cpu().numpy())
    #
    # preds = np.concatenate(preds, axis=0).reshape(-1, 1)
    # labels = np.concatenate(labels, axis=0).reshape(-1, 1)
    # acc = np.sum(preds == labels) / len(labels)

    # print("Epoch: {}, loss: {}".format(epo, np.mean(epo_loss)))
    logger.info(
        "Epoch {} Training loss: {}".format(
            epo, np.sum(epo_loss) / train_size))
    test_loss, acc = eval_model(resnet, test_dsld)
    logger.info(
        "Epoch {} Test loss: {}, ACC: {}".format(
            epo, test_loss / test_size, acc))






