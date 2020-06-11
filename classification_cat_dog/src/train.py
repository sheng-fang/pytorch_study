"""Functions for training a model

"""
import os
import pathlib
import copy
import time

import numpy as np
import pandas as pd
import torch
import torchvision
import matplotlib.pyplot as plt

from classification_cat_dog.src import data_pipeline
from classification_cat_dog.src import modelling


def train_model(nb_epoch, train_dsld, model, opt, criterion, scheduler=None,
                device=None):
    """
    Function to train a model.
    Args:
        nb_epoch: number of epoch to train
        train_dsld: dataset loader for training
        model: model to optimize
        opt: optimizer

    Returns:
        model: a trained model

    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    st_start = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(nb_epoch):
        print("Epcoh {}/{}".format(epoch, nb_epoch))
        print("_" * 10)
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in train_dsld:
                inputs = inputs.to(device)
                labels = labels.to(device)

                opt.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    pred_y = model(inputs)
                    _, preds = torch.max(pred_y, 1)


