#!usr/bin/env python  
#-*- coding:utf-8 _*- 
"""
@version: python3.6
@author: ikkyu-wen
@contact: wenruichn@gmail.com
@time: 2019-08-15 14:25
公众号：AI成长社
知乎：https://www.zhihu.com/people/qlmx-61/columns
"""
import argparse
from build_net import model_names

# Parse arguments
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# Datasets
parser.add_argument('-train', '--trainroot', default='data/new_shu_label.txt', type=str) #new_shu_label
parser.add_argument('-val', '--valroot', default='data/val1.txt', type=str)

parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--num-classes', default=43, type=int, metavar='N',
                    help='number of classfication of image')
parser.add_argument('--image-size', default=288, type=int, metavar='N',
                    help='the train image size')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=64, type=int, metavar='N',
                    help='train batchsize (default: 256)')
parser.add_argument('--test-batch', default=32, type=int, metavar='N',
                    help='test batchsize (default: 200)')
parser.add_argument('--optimizer', default='sgd',
                         choices=['sgd', 'rmsprop', 'adam', 'AdaBound', 'radam'], metavar='N',
                         help='optimizer (default=sgd)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate，1e-2， 1e-4, 0.001')
parser.add_argument('--lr-fc-times', '--lft', default=5, type=int,
                    metavar='LR', help='initial model last layer rate')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--schedule', type=int, nargs='+', default=[30, 50, 60],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--no_nesterov', dest='nesterov',
                         action='store_false',
                         help='do not use Nesterov momentum')
parser.add_argument('--alpha', default=0.99, type=float, metavar='M',
                         help='alpha for ')
parser.add_argument('--beta1', default=0.9, type=float, metavar='M',
                         help='beta1 for Adam (default: 0.9)')
parser.add_argument('--beta2', default=0.999, type=float, metavar='M',
                         help='beta2 for Adam (default: 0.999)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--final-lr', '--fl', default=1e-3,type=float,
                    metavar='W', help='weight decay (default: 1e-3)')
# Checkpoints
parser.add_argument('-c', '--checkpoint', default='/data0/search/qlmx/clover/garbage/res_16_288_last1', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnext101_32x16d_wsl',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnext101_32x8d, pnasnet5large)')
parser.add_argument('--depth', type=int, default=29, help='Model depth.')
parser.add_argument('--cardinality', type=int, default=32, help='ResNet cardinality (group).')
parser.add_argument('--base-width', type=int, default=4, help='ResNet base width.')
parser.add_argument('--widen-factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
#Device options
parser.add_argument('--gpu-id', default='0, 1, 2, 3', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()
