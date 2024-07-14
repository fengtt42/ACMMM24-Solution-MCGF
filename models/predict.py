# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import numpy as np
from torchvision import datasets, transforms
import time
import os
import yaml
import math
import scipy.io
from model import ft_net, two_view_net, three_view_net
from utils import load_network
from image_folder import customData, customData_one, customData_style, ImageFolder_iaa
import imgaug.augmenters as iaa
import pandas as pd
try:
    from apex.fp16_utils import *
except ImportError: # will be 3.x series
    print('This is not an error. If you want to use low precision, i.e., fp16, please install the apex with cuda support (https://github.com/NVIDIA/apex) and update pytorch to 1.0')
######################################################################
# Options
# --------
#
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--which_epoch',default='last', type=str, help='0,1,2,3...or last')
parser.add_argument('--test_dir',default='./data/test',type=str, help='./test_data')
parser.add_argument('--name', default='three_view_long_share_d0.75_256_s1_google', type=str, help='save model path')
parser.add_argument('--pool', default='avg', type=str, help='avg|max')
parser.add_argument('--style', default='none', type=str, help='select image style: e.g. night, nightfall, NightLight, shadow, StrongLight, all')
parser.add_argument('--batchsize', default=64, type=int, help='batchsize')
parser.add_argument('--h', default=256, type=int, help='height')
parser.add_argument('--w', default=256, type=int, help='width')
parser.add_argument('--views', default=2, type=int, help='views')
parser.add_argument('--pad', default=0, type=int, help='padding')
parser.add_argument('--use_dense', action='store_true', help='use densenet121' )
parser.add_argument('--LPN', action='store_true', help='use LPN' )
parser.add_argument('--multi', action='store_true', help='use multiple query' )
parser.add_argument('--fp16', action='store_true', help='use fp16.' )
parser.add_argument('--scale_test', action='store_true', help='scale test' )
parser.add_argument('--iaa', action='store_true', help='iaa image augmentation' )
parser.add_argument('--ms',default='1', type=str,help='multiple_scale: e.g. 1 1,1.1  1,1.1,1.2')

opt = parser.parse_args()

######################################################################
# load the training config
# --------
#
config_path = os.path.join('./model',opt.name,'opts.yaml')
with open(config_path, 'r') as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)
opt.fp16 = config['fp16']
opt.use_dense = config['use_dense']
opt.use_NAS = config['use_NAS']
opt.stride = config['stride']
opt.views = config['views']
opt.LPN = config['LPN']
opt.block = config['block']
scale_test = opt.scale_test
style = opt.style
if 'h' in config:
    opt.h = config['h']
    opt.w = config['w']
print('------------------------------', opt.h)
if 'nclasses' in config: # tp compatible with old config files
    opt.nclasses = config['nclasses']
else:
    opt.nclasses = 729

str_ids = opt.gpu_ids.split(',')
name = opt.name
test_dir = opt.test_dir

gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >=0:
        gpu_ids.append(id)

str_ms = opt.ms.split(',')
ms = []
for s in str_ms:
    s_f = float(s)
    ms.append(math.sqrt(s_f))

# set gpu ids
if len(gpu_ids)>0:
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str,gpu_ids))
    # torch.cuda.set_device(gpu_ids[0])
    cudnn.benchmark = True

use_gpu = torch.cuda.is_available()

######################################################################
# Load Data
# ---------
#
class CustomImageFolder(datasets.ImageFolder):
    def __init__(self, root, file_list, transform=None, target_transform=None):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.samples = self._make_dataset(file_list)
        # print(self.samples)

    def _make_dataset(self, file_list):
        data = []
        for line in file_list:
            path = os.path.join(self.root, "query_drone_160k_wx_24", line)
            item = (path, int(0))
            data.append(item)
        return data

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

def datacollection(data_dir, query_name, gallery_name):
    name_rank = []
    with open("query_drone_test.txt", "r") as f:
        for txt in f.readlines():
            name_rank.append(txt[:-1])
    data_transforms = transforms.Compose([
        transforms.Resize((opt.h, opt.w), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image_datasets = {}
    image_datasets[gallery_name] = datasets.ImageFolder(os.path.join(data_dir, gallery_name),
                                                               data_transforms)
    image_datasets[query_name] = CustomImageFolder(os.path.join(data_dir, query_name), name_rank,
                                                      data_transforms)
    # opt.batchsize
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=128,
                                             shuffle=False, num_workers=8) for x in [gallery_name, query_name]}
    with open('query_drone_test.txt', 'r') as f:
        order = [line.strip() for line in f.readlines()]
    image_datasets[query_name].imgs = sorted(image_datasets[query_name].imgs, key=lambda x: order.index(x[0].split("/")[-1]))
    return image_datasets, dataloaders

######################################################################
# Extract feature
# ----------------------
#
def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip

def which_view(name):
    if 'satellite' in name:
        return 1
    elif 'street' in name:
        return 2
    elif 'drone' in name:
        return 3
    else:
        print('unknown view')
    return -1

def extract_feature(model, dataloaders, view_index = 1):
    features = torch.FloatTensor()
    count = 0
    for data in dataloaders:
        img, label = data
        n, c, h, w = img.size()
        count += n
        ff = torch.FloatTensor(n, 512).zero_().cuda()
        if opt.LPN:
            # ff = torch.FloatTensor(n,2048,6).zero_().cuda()
            ff = torch.FloatTensor(n, 512, opt.block).zero_().cuda()
        for i in range(2):
            if(i==1):
                img = fliplr(img)
            input_img = Variable(img.cuda())
            for scale in ms:
                if scale != 1:
                    # bicubic is only  available in pytorch>= 1.1
                    input_img = nn.functional.interpolate(input_img, scale_factor=scale, mode='bilinear', align_corners=False)
                if opt.views ==2:
                    if view_index == 1:
                        outputs, _ = model(input_img, None)
                    elif view_index ==2:
                        _, outputs = model(None, input_img)
                elif opt.views ==3:
                    if view_index == 1:
                        outputs, _, _ = model(input_img, None, None)
                    elif view_index ==2:
                        _, outputs, _ = model(None, input_img, None)
                    elif view_index ==3:
                        _, _, outputs = model(None, None, input_img)
                ff += outputs
        # norm feature
        if opt.LPN:
            # feature size (n,2048,6)
            # 1. To treat every part equally, I calculate the norm for every 2048-dim part feature.
            # 2. To keep the cosine score==1, sqrt(6) is added to norm the whole feature (2048*6).
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True) * np.sqrt(opt.block)
            ff = ff.div(fnorm.expand_as(ff))
            ff = ff.view(ff.size(0), -1)
        else:
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            ff = ff.div(fnorm.expand_as(ff))

        features = torch.cat((features,ff.data.cpu()), 0)
    return features

def get_id(img_path):
    camera_id = []
    labels = []
    paths = []
    for path, v in img_path:
        # print(path, v)
        folder_name = os.path.basename(os.path.dirname(path))
        labels.append(int(folder_name))
        paths.append(path)
    return labels, paths

def get_result(query_feature, gallery_feature, query_img_list):
    result = {}
    for i in range(len(query_img_list)):

        query = query_feature[i].view(-1, 1)
        score = torch.mm(gallery_feature, query)
        score = score.squeeze(1).cpu()
        index = np.argsort(score.numpy())
        index = index[::-1].tolist()
        max_score_list = index[0:10]
        query_img = query_img_list[i][0]
        most_correlative_img = []
        for index in max_score_list:
            most_correlative_img.append(gallery_img_list[index][0])
        result[query_img] = most_correlative_img
    matching_table = pd.DataFrame(result)
    matching_table.to_csv("result.csv")

def get_answer(result):
    table = pd.read_csv(result, index_col=0)
    result = {}
    for i in table:
        result[i.split("/")[-1]] = [k.split("/")[-1].split(".")[0] for k in list(table[i])]

    with open("query_drone_test.txt", "r") as f:
        txt = f.readlines()
        f.close()
    txt = [i.split("\n")[0] for i in txt]
    with open("answer.txt", "w") as p:
        for t in txt:
            p.write(' '.join(result[t]))
            p.write("\n")

######################################################################
# Load Collected data Trained model
# ----------
#
model, _, epoch = load_network(opt.name, opt)
model.classifier.classifier = nn.Sequential()
model = model.eval()
if use_gpu:
    model = model.cuda()

######################################################################
# predict
# ----------
#
if __name__ == "__main__":
    print('------------------Processing Images----------------------')
    data_dir = '/home/ftt/UAVM/datasets'
    query_name = 'query_drone'
    gallery_name = 'gallery_satellite'
    image_datasets, dataloaders = datacollection(data_dir, query_name, gallery_name)

    print('------------------Extract feature----------------------')
    since = time.time()
    which_query = which_view(query_name)
    which_gallery = which_view(gallery_name)
    print('%d -> %d:' % (which_query, which_gallery))

    with torch.no_grad():
        query_img_list = image_datasets[query_name].imgs
        gallery_img_list = image_datasets[gallery_name].imgs
        query_feature = extract_feature(model, dataloaders[query_name], which_query)
        gallery_feature = extract_feature(model, dataloaders[gallery_name], which_gallery)

    time_elapsed = time.time() - since
    print('Test complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    # Save to Matlab for check
    result = {'gallery_f': gallery_feature.numpy(), 'query_f': query_feature.numpy()}
    scipy.io.savemat('pytorch_predict.mat', result)

    print('------------------Getting Result----------------------')
    get_result(query_feature, gallery_feature, query_img_list)
    get_answer("result.csv")
