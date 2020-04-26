#!usr/bin/env python  
#-*- coding:utf-8 _*- 
"""
@version: python3.6
@author: QLMX
@contact: wenruichn@gmail.com
@time: 2019-08-15 14:30
"""
from torch import nn
import torchvision.models as models
import models as customized_models

# Models
default_model_names = sorted(name for name in models.__dict__
                             if name.islower() and not name.startswith("__")
                             and callable(models.__dict__[name]))

customized_models_names = sorted(name for name in customized_models.__dict__
                                 if not name.startswith("__")
                                 and callable(customized_models.__dict__[name]))

for name in customized_models.__dict__:
    if not name.startswith("__") and callable(customized_models.__dict__[name]):
        models.__dict__[name] = customized_models.__dict__[name]

model_names = default_model_names + customized_models_names


def make_model(args):
    print("=> creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch](progress=True)
    model.fc = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(2048, args.num_classes)
    )
    return model

if __name__=='__main__':
    all_model = sorted(name for name in models.__dict__ if not name.startswith("__"))
    print(all_model)
