import logging
import torch
import torch.utils.data
import torch.nn as nn
import numpy as np
from utils.dataloaders import (full_path_loader, full_test_loader, CDDloader)
from utils.metrics import jaccard_loss, dice_loss
from utils.losses import hybrid_loss,cross_entropy
# from utils.losses import hybrid_loss, cross_entropy
from models.Net import Net
logging.basicConfig(level=logging.INFO)

def initialize_metrics():
    """Generates a dictionary of metrics with metrics as keys
       and empty lists as values

    Returns
    -------
    dict
        a dictionary of metrics

    """
    metrics = {
        'cd_losses': [],
        'cd_corrects': [],
        'cd_precisions': [],
        'cd_recalls': [],
        'cd_f1scores': [],
        'learning_rate': [],
    }

    return metrics


def get_mean_metrics(metric_dict):
    """takes a dictionary of lists for metrics and returns dict of mean values

    Parameters
    ----------
    metric_dict : dict
        A dictionary of metrics

    Returns
    -------
    dict
        dict of floats that reflect mean metric value

    """
    return {k: np.mean(v) for k, v in metric_dict.items()}


def set_metrics(metric_dict, cd_loss, cd_corrects, cd_report, lr):
    """Updates metric dict with batch metrics

    Parameters
    ----------
    metric_dict : dict
        dict of metrics
    cd_loss : dict(?)
        loss value
    cd_corrects : dict(?)
        number of correct results (to generate accuracy
    cd_report : list
        precision, recall, f1 values

    Returns
    -------
    dict
        dict of  updated metrics


    """
    metric_dict['cd_losses'].append(cd_loss.item())
    metric_dict['cd_corrects'].append(cd_corrects.item())
    metric_dict['cd_precisions'].append(cd_report[0])
    metric_dict['cd_recalls'].append(cd_report[1])
    metric_dict['cd_f1scores'].append(cd_report[2])
    metric_dict['learning_rate'].append(lr)

    return metric_dict

def set_test_metrics(metric_dict, cd_corrects, cd_report):

    metric_dict['cd_corrects'].append(cd_corrects.item())
    metric_dict['cd_precisions'].append(cd_report[0])
    metric_dict['cd_recalls'].append(cd_report[1])
    metric_dict['cd_f1scores'].append(cd_report[2])

    return metric_dict


def get_loaders(opt):


    logging.info('STARTING Dataset Creation')

    train_full_load, val_full_load = full_path_loader(opt.dataset_dir)


    train_dataset = CDDloader(train_full_load, aug=opt.augmentation)
    print(len(train_dataset))
    val_dataset = CDDloader(val_full_load, aug=False)
    print(len(val_dataset))
    logging.info('STARTING Dataloading')

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=opt.batch_size,
                                               shuffle=True,
                                               num_workers=opt.num_workers)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=opt.batch_size,
                                             shuffle=False,
                                             num_workers=opt.num_workers)
    return train_loader, val_loader

def get_test_loaders(opt, batch_size=None):

    if not batch_size:
        batch_size = opt.batch_size

    logging.info('STARTING Dataset Creation')

    test_full_load = full_test_loader(opt.dataset_dir)

    test_dataset = CDDloader(test_full_load, aug=False)

    logging.info('STARTING Dataloading')


    test_loader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=opt.num_workers)
    return test_loader


def get_criterion(opt):
    """get the user selected loss function

    Parameters
    ----------
    opt : dict
        Dictionary of options/flags

    Returns
    -------
    method
        loss function

    """
    if opt.loss_function == 'hybrid':
        criterion = hybrid_loss
    if opt.loss_function == 'bce':
        criterion = nn.CrossEntropyLoss()  #交叉熵损失函数，应该是可以分三类的
    if opt.loss_function == 'dice':
        criterion = dice_loss
    if opt.loss_function == 'jaccard':
        criterion = jaccard_loss
    if opt.loss_function == 'mce':
        criterion = cross_entropy

    return criterion


def load_model(opt, device):
    """Load the model

    Parameters
    ----------
    opt : dict
        User specified flags/options
    device : string
        device on which to train model

    """
    # device_ids = list(range(opt.num_gpus))
    num_class = 5
    model = Net(num_class).to(device)
    print(model)
    # model = Siam_NestedUNet_Conc(opt.num_channel, 5).to(device)
    # if pt_file 
    # model = torch.load('./tmp-LEVIRnew-threeclass/checkpoint_epoch_14.pt')
    # model = SNUNet_ECAM(opt.num_channel, 5).to(device)
    # model = nn.DataParallel(model, device_ids=device_ids)
    return model
