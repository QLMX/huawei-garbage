#!usr/bin/env python  
#-*- coding:utf-8 _*- 
"""
@version: python3.6
@author: QLMX
@contact: wenruichn@gmail.com
@time: 2019-08-20 00:55
"""
import torch.nn as nn
import cv2
import torch
device=torch.device("cuda")
import os
import urllib.parse as urlparse
import requests
import torch

__all__ = ['GetEncoder', 'GetPreTrainedModel', 'load_pretrained', 'l2_norm']


#修改模型以进行特征提取
def GetEncoder(model):
    layerName,layer=list(model.named_children())[-1]
    exec("model."+layerName+"=nn.Linear(layer.in_features,layer.in_features)")
    exec("torch.nn.init.eye_(model."+layerName+".weight)")
    for param in model.parameters():
        param.requires_grad=False
    return model,layer.in_features

#修改模型以进行微调，n_ZeroChild和n_ZeroLayer用来设置参数固定层，当children为Sequential时使用n_ZeroLayer，可对其内部进行设置
def GetPreTrainedModel(model,n_Output,n_ZeroChild,n_ZeroLayer=None):
    for i,children in enumerate(model.children()):
        if i==n_ZeroChild:
            if n_ZeroLayer is not None:
                for j,layer in enumerate(children):
                    if j==n_ZeroLayer:
                        break
                    for param in layer.parameters():
                        param.requires_grad=False
            break
        for param in children.parameters():
            param.requires_grad=False
    layerName,layer=list(model.named_children())[-1]
    exec("model."+layerName+"=nn.Linear(layer.in_features,"+str(n_Output)+")")
    return model


class StackNet2(nn.Module):
    def __init__(self,models,n_Target):
        super(StackNet,self).__init__()
        self.models=models
        n_Out=0
        for i,(model,scale_In,n_ZeroChild,n_ZeroLayer) in enumerate(self.models):
            for j,children in enumerate(model.children()):
                if j==n_ZeroChild:
                    if n_ZeroLayer is not None:
                        for k,layer in enumerate(children):
                            if k==n_ZeroLayer:
                                break
                            for param in layer.parameters():
                                param.requires_grad=False
                    break

            layerName,layer=list(model.named_children())[-1]
            n_Out+=layer.in_features
            exec("model."+layerName+"=nn.Linear(layer.in_features,layer.in_features)")
            exec("torch.nn.init.eye_(model."+layerName+".weight)")
            exec("layer=model."+layerName)
            for param in layer.parameters():
                param.requires_grad=False
            exec("self.model"+str(i)+"=model")
        self.fc=nn.Linear(n_Out,n_Target)
    def forward(self,x):
        feature=[]
        for model,scale_In,_,_ in self.models:
            feature.append(model(x))
        feature=torch.cat(feature,dim=1)
        return self.fc(feature)


def _download_file_from_google_drive(fid, dest):
    def _get_confirm_token(res):
        for k, v in res.cookies.items():
            if k.startswith('download_warning'): return v
        return None

    def _save_response_content(res, dest):
        CHUNK_SIZE = 32768
        with open(dest, "wb") as f:
            for chunk in res.iter_content(CHUNK_SIZE):
                if chunk: f.write(chunk)

    URL = "https://docs.google.com/uc?export=download"
    sess = requests.Session()
    res = sess.get(URL, params={'id': fid}, stream=True)
    token = _get_confirm_token(res)

    if token:
        params = {'id': fid, 'confirm': token}
        res = sess.get(URL, params=params, stream=True)
    _save_response_content(res, dest)


def _load_url(url, dest):
    if os.path.isfile(dest) and os.path.exists(dest): return dest
    print('[INFO]: Downloading weights...')
    fid = urlparse.parse_qs(urlparse.urlparse(url).query)['id'][0]
    _download_file_from_google_drive(fid, dest)
    return dest


def load_pretrained(m, meta, dest, pretrained=False):
    if pretrained:
        if len(meta) == 0:
            print('[INFO]: Pretrained model not available')
            return m
        if dest is None: dest = meta[0]
        else:
            dest = dest + '/' + meta[0]
        print(dest)
        m.load_state_dict(torch.load(_load_url(meta[1], dest)))
    return m

def l2_norm(input,axis=1):
    norm = torch.norm(input,2,axis,True)
    output = torch.div(input, norm)
    return output