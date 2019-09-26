'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import errno
import os
import sys
import time
import math

import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import torch
import shutil
import adabound
from utils.radam import RAdam, AdamW
import torchvision.transforms as transforms



__all__ = ['get_mean_and_std', 'init_params', 'mkdir_p', 'AverageMeter', 'get_optimizer', 'save_checkpoint']


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = trainloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)

    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)

def mkdir_p(path):
    '''make dir if not exist'''
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def get_optimizer(model, args):
    parameters = []
    for name, param in model.named_parameters():
        if 'fc' in name or 'class' in name or 'last_linear' in name or 'ca' in name or 'sa' in name:
            parameters.append({'params': param, 'lr': args.lr * args.lr_fc_times})
        else:
            parameters.append({'params': param, 'lr': args.lr})

    if args.optimizer == 'sgd':
        return torch.optim.SGD(parameters,
                            # model.parameters(),
                               args.lr,
                               momentum=args.momentum, nesterov=args.nesterov,
                               weight_decay=args.weight_decay)
    elif args.optimizer == 'rmsprop':
        return torch.optim.RMSprop(parameters,
                                # model.parameters(),
                                   args.lr,
                                   alpha=args.alpha,
                                   weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        return torch.optim.Adam(parameters,
                                # model.parameters(),
                                args.lr,
                                betas=(args.beta1, args.beta2),
                                weight_decay=args.weight_decay)
    elif args.optimizer == 'AdaBound':
        return adabound.AdaBound(parameters,
                                # model.parameters(),
                                lr=args.lr, final_lr=args.final_lr)
    elif args.optimizer == 'radam':
        return RAdam(parameters, lr=args.lr, betas=(args.beta1, args.beta2),
                          weight_decay=args.weight_decay)

    else:
        raise NotImplementedError


def save_checkpoint(state, is_best, single=True, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    if single:
        fold = ''
    else:
        fold = str(state['fold']) + '_'
    cur_name = 'checkpoint.pth.tar'
    filepath = os.path.join(checkpoint, fold + cur_name)
    curpath = os.path.join(checkpoint, fold + 'model_cur.pth')

    torch.save(state, filepath)
    torch.save(state['state_dict'], curpath)

    if is_best and state['epoch'] >= 5:
        model_name = 'model_' + str(state['epoch']) + '_' + str(int(round(state['train_acc']*100, 0))) + '_' + str(int(round(state['acc']*100, 0))) + '.pth'
        model_path = os.path.join(checkpoint, fold + model_name)
        torch.save(state['state_dict'], model_path)


def save_checkpoint2(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    # best_model = '/application/search/qlmx/clover/garbage/code/image_classfication/predict/'
    fold = str(state['fold']) + '_'
    filepath = os.path.join(checkpoint, fold + filename)
    model_path = os.path.join(checkpoint, fold + 'model_cur.pth')

    torch.save(state, filepath)
    torch.save(state['state_dict'], model_path)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, fold + 'model_best.pth.tar'))
        shutil.copyfile(model_path, os.path.join(checkpoint, fold + 'model_best.pth'))
