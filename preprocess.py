#!usr/bin/env python  
#-*- coding:utf-8 _*- 
"""
@version: python3.6
@author: QLMX
@contact: wenruichn@gmail.com
@time: 2019-08-14 11:16
公众号：AI成长社
知乎：https://www.zhihu.com/people/qlmx-61/columns
"""
from glob import glob
import os
import codecs
import random
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold


base_path = 'data/'
data_path = base_path + 'garbage_classify/train_data'

label_files = glob(os.path.join(data_path, '*.txt'))
img_paths = []
labels = []
result = []
label_dict = {}
data_dict = {}

for index, file_path in enumerate(label_files):
    with codecs.open(file_path, 'r', 'utf-8') as f:
        line = f.readline()
    line_split = line.strip().split(', ')
    if len(line_split) != 2:
        print('%s contain error lable' % os.path.basename(file_path))
        continue
    img_name = line_split[0]
    label = int(line_split[1])
    img_paths.append(os.path.join(data_path, img_name))
    labels.append(label)
    result.append(os.path.join(data_path, img_name) + ',' + str(label))
    label_dict[label] = label_dict.get(label, 0) + 1
    if label not in data_dict:
        data_dict[label] = []
    data_dict[label].append(os.path.join(data_path, img_name) + ',' + str(label))

data_path_add = base_path + 'garbage_classify_v3'
label_files_add = glob(os.path.join(data_path_add, '*.txt'))

for index, file_path in enumerate(label_files_add):
    with codecs.open(file_path, 'r', 'utf-8') as f:
        line = f.readline()
    line_split = line.strip().split(', ')
    if len(line_split) != 2:
        print('%s contain error lable' % os.path.basename(file_path))
        continue
    img_name = line_split[0]
    label = int(line_split[1])
    img_paths.append(os.path.join(data_path_add, img_name))
    labels.append(label)
    result.append(os.path.join(data_path_add, img_name) + ',' + str(label))
    label_dict[label] = label_dict.get(label, 0) + 1
    if label not in data_dict:
        data_dict[label] = []
    data_dict[label].append(os.path.join(data_path_add, img_name) + ',' + str(label))


folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=2019)
for fold_, (trn_idx, val_idx) in enumerate(folds.split(result, labels)):
    train_data = list(np.array(result)[trn_idx])
    val_data = list(np.array(result)[val_idx])

print(len(train_data), len(val_data))

with open(base_path + 'train1.txt', 'w') as f1:
    for item in train_data:
        f1.write(item + '\n')

with open(base_path + 'val1.txt', 'w') as f2:
    for item in val_data:
        f2.write(item + '\n')


from PIL import Image

###predata 2
all_data = []
train = []
val = []
rate = 0.9
import cv2
from tqdm import tqdm

error_list = ['data/additional_train_data/38/242.jpg',
              'data/additional_train_data/34/79.jpg',
              'data/additional_train_data/27/55.jpg'
              'data/new/8/0.jpg'
              ]

data_path = base_path + 'new/'
for i in range(40):
    na_item = []
    img_files = glob(os.path.join(data_path, str(i), '*.jpg'))
    for item in tqdm(img_files):
        ii = cv2.imread(item)
        if item not in error_list:

            jj = Image.open(item).layers
            if jj == 1:
                print(item)
            all_data.append(item + ',' + str(i))
            na_item.append(item + ',' + str(i))
    random.shuffle(na_item)
    train.extend(na_item[ : int(len(na_item)*rate)])
    val.extend(na_item[int(len(na_item)*rate):])
print(len(train), len(val))

random.shuffle(all_data)
random.shuffle(train)
random.shuffle(val)

print(len(all_data))

old = []
with open(base_path + 'train1.txt', 'r') as f:
    for i in f.readlines():
        old.append(i.strip())
for i in all_data:
    img_path, label = i.strip().split(',')

all_data.extend(old)
print(len(all_data))
random.shuffle(all_data)

with open(base_path + 'new_shu_label.txt', 'w') as f1:
    for item in all_data:
        f1.write(item + '\n')