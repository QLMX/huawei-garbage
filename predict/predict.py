#!usr/bin/env python
#-*- coding:utf-8 _*-
"""
@version: python3.6
@author: QLMX
@contact: wenruichn@gmail.com
@time: 2019-08-14 16:07
"""
import torchvision.transforms as transforms
import torch
from PIL import Image
from collections import OrderedDict
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
from torch import nn
import os, time
import torchvision.models as models
from resnetxt_wsl import resnext101_32x8d_wsl, resnext101_32x16d_wsl, resnext101_32x32d_wsl
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

args = {}
args['arch'] = 'resnext101_32x16d_wsl'
args['pretrained'] = False
args['num_classes'] = 43
args['image_size'] = 320


class classfication_service():
    def __init__(self, model_path):
        self.model = self.build_model(model_path)
        self.pre_img = self.preprocess_img()
        self.model.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.label_id_name_dict = \
            {
                "0": "其他垃圾/一次性快餐盒",
                "1": "其他垃圾/污损塑料",
                "2": "其他垃圾/烟蒂",
                "3": "其他垃圾/牙签",
                "4": "其他垃圾/破碎花盆及碟碗",
                "5": "其他垃圾/竹筷",
                "6": "厨余垃圾/剩饭剩菜",
                "7": "厨余垃圾/大骨头",
                "8": "厨余垃圾/水果果皮",
                "9": "厨余垃圾/水果果肉",
                "10": "厨余垃圾/茶叶渣",
                "11": "厨余垃圾/菜叶菜根",
                "12": "厨余垃圾/蛋壳",
                "13": "厨余垃圾/鱼骨",
                "14": "可回收物/充电宝",
                "15": "可回收物/包",
                "16": "可回收物/化妆品瓶",
                "17": "可回收物/塑料玩具",
                "18": "可回收物/塑料碗盆",
                "19": "可回收物/塑料衣架",
                "20": "可回收物/快递纸袋",
                "21": "可回收物/插头电线",
                "22": "可回收物/旧衣服",
                "23": "可回收物/易拉罐",
                "24": "可回收物/枕头",
                "25": "可回收物/毛绒玩具",
                "26": "可回收物/洗发水瓶",
                "27": "可回收物/玻璃杯",
                "28": "可回收物/皮鞋",
                "29": "可回收物/砧板",
                "30": "可回收物/纸板箱",
                "31": "可回收物/调料瓶",
                "32": "可回收物/酒瓶",
                "33": "可回收物/金属食品罐",
                "34": "可回收物/锅",
                "35": "可回收物/食用油桶",
                "36": "可回收物/饮料瓶",
                "37": "有害垃圾/干电池",
                "38": "有害垃圾/软膏",
                "39": "有害垃圾/过期药物",
                "40": "可回收物/毛巾",
                "41": "可回收物/饮料盒",
                "42": "可回收物/纸袋"
            }

    def build_model(self, model_path):
        if args['arch'] == 'resnext101_32x16d_wsl':
            model = resnext101_32x16d_wsl(pretrained=False, progress=False)
        if args['arch'] == 'resnext101_32x8d':
            model = models.__dict__[args['arch']]()
        elif args['arch'] == 'efficientnet-b7':
            model = EfficientNet.from_name(args['arch'])

        layerName, layer = list(model.named_children())[-1]
        exec("model." + layerName + "=nn.Linear(layer.in_features," + str(args['num_classes']) + ")")

        if torch.cuda.is_available():
            modelState = torch.load(model_path)
            model.load_state_dict(modelState)
            model = model.cuda()
        else:
            modelState = torch.load(model_path, map_location='cpu')
            model.load_state_dict(modelState)
        return model

    def preprocess_img(self):
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        infer_transformation = transforms.Compose([
            Resize((args['image_size'], args['image_size'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        return infer_transformation

    def _preprocess(self, data):
        preprocessed_data = {}
        for k, v in data.items():
            for file_name, file_content in v.items():
                img = Image.open(file_content)
                img = self.pre_img(img)
                preprocessed_data[k] = img
        return preprocessed_data

    def _inference(self, data):
        """
        model inference function
        Here are a inference example of resnet, if you use another model, please modify this function
        """
        img = data['input_img']
        img = img.unsqueeze(0)
        img = img.to(self.device)
        with torch.no_grad():
            pred_score = self.model(img)

        if pred_score is not None:
            _, pred_label = torch.max(pred_score.data, 1)
            result = {'result': self.label_id_name_dict[str(pred_label[0].item())]}
        else:
            result = {'result': 'predict score is None'}

        return result

    def _postprocess(self, data):
        return data


class Resize(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        ratio = self.size[0] / self.size[1]
        w, h = img.size
        if w / h < ratio:
            t = int(h * ratio)
            w_padding = (t - w) // 2
            img = img.crop((-w_padding, 0, w+w_padding, h))
        else:
            t = int(w / ratio)
            h_padding = (t - h) // 2
            img = img.crop((0, -h_padding, w, h+h_padding))

        img = img.resize(self.size, self.interpolation)

        return img

if __name__ == '__main__':
    model_path = '/Users/QLMX/Documents/' + 'model_19_9992_9590.pth'
    infer = classfication_service(model_path)
    input_dir = '/Users/QLMX/Downloads/garbage_classify_v3_select_100'
    files = os.listdir(input_dir)
    t1 = int(time.time()*1000)
    for file_name in files:
        file_path = os.path.join(input_dir, file_name)
        img = Image.open(file_path)

        img = infer.pre_img(img)
        tt1 = int(time.time() * 1000)
        result = infer._inference({'input_img': img})
        tt2 = int(time.time() * 1000)
        print((tt2 - tt1) / 100)
    t2 = int(time.time()*1000)
    print((t2 - t1)/100)

