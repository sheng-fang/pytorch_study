import logging

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def get_logger(logger_file, file_logger_level, logger_name=None,
               add_console_logger=True, console_logger_level=None):
    """
    Generate a logger
    Args:
        logger_file: path to logger file
        file_logger_level: logger level for information to the file
        logger_name: name for the logger
        add_console_logger: config a console logger or not
        console_logger_level: console logger level

    Returns:

    """
    # create logger
    if logger_name is None:
        logger_name = logger_file
    logger = logging.getLogger(logger_name)
    logger.setLevel(file_logger_level)

    # create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    if add_console_logger:
        # create console handler and set level to debug
        if console_logger_level is None:
            console_logger_level = file_logger_level
        ch = logging.StreamHandler()
        ch.setLevel(console_logger_level)
        # add formatter to ch
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    fh = logging.FileHandler(logger_file)
    logger.addHandler(fh)

    return logger


def calculate_accuracy(threshold, sims, actual_issame):
    predict_issame = np.greater(sims, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(
        np.logical_and(np.logical_not(predict_issame),
                       np.logical_not(actual_issame)
                       )
    )
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / sims.size
    return tpr, fpr, acc


def calculate_roc(thresholds, sims, lables):
    tprs = []
    fprs = []
    for threshold in thresholds:
        tmp_tpr, tmp_fpr, _ = calculate_accuracy(threshold, sims, lables)
        tprs.append(tmp_tpr)
        fprs.append(tmp_fpr)

    return np.asarray(fprs), np.asarray(tprs)


def find_lr(model, dsloader, optim, criterion, device, lr_list=None,
            lr_init=None, lr_end=None, lr_nb=None):
    if lr_list is None:
        cond = (lr_init is not None
                and lr_end is not None
                and lr_nb is not None)
        assert cond, "Input lr_init, lr_end, and lr_nb, when lr_list is None"
        log_init = np.log(lr_init)
        log_end = np.log(lr_end)
        lr_list = np.arange(log_init, log_end, (log_end - log_init) / lr_nb)
        lr_list = np.power(np.e, lr_list)
    init_stat = model.state_dict().copy()
    model.to(device)
    model.train()
    loss_list = []
    for lr in lr_list:
        for param in optim.param_groups:
            param["lr"] = lr
        tmp_loss = []
        for data, label in tqdm(iter(dsloader)):
            data = data.to(device)
            label = label.to(device)

            optim.zero_grad()
            pred = model(data)
            loss = criterion(pred, label)

            loss.backward()
            optim.step()

            tmp_loss.append(loss.item())
        loss_list.append(np.mean(tmp_loss))

    model.load_state_dict(init_stat)

    return model, loss_list, lr_list
