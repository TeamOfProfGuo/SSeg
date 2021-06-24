###########################################################################
# Created by: Hang Zhang
# Email: zhang.hang@rutgers.edu
# Copyright (c) 2017
###########################################################################

import os, sys
BASE_DIR = os.path.dirname(os.path.dirname(os.getcwd()))
sys.path.append(BASE_DIR)
import copy
import yaml
import logging
import argparse
import numpy as np
from tqdm import tqdm
from addict import Dict

import torch
import torch.nn as nn
from torch.utils import data
from tensorboardX import SummaryWriter
import torchvision.transforms as transform
from torch.nn.parallel.scatter_gather import gather

import encoding.utils as utils
from encoding.nn import SegmentationLosses, SyncBatchNorm
from encoding.parallel import DataParallelModel, DataParallelCriterion
from encoding.datasets import get_dataset
from encoding.models import get_segmentation_model

BASE_DIR = '.'
CONFIG_PATH = 'experiments/attlinknet/results/config.yaml'
SMY_PATH = os.path.dirname(CONFIG_PATH)
GPUS = [0,1]


# =====================  setup  ======================

# configuration
args = Dict(yaml.safe_load(open(CONFIG_PATH)))
args.cuda = (args.use_cuda and torch.cuda.is_available())
torch.manual_seed(args.seed)
args.batch_size = 2

# ================= trainer init  ======================
# data transforms
input_transform = transform.Compose([
    transform.ToTensor(),  # convert RGB [0,255] to FloatTensor in range [0, 1]
    transform.Normalize([.485, .456, .406], [.229, .224, .225])])   # mean and std based on imageNet
dep_transform = transform.Compose([
    transform.ToTensor(),
    transform.Normalize(mean=[0.2798], std=[0.1387])  # mean and std for depth
])
# dataset
data_kwargs = {'transform': input_transform, 'dep_transform':dep_transform,
               'base_size': args.base_size, 'crop_size': args.crop_size}
trainset = get_dataset(args.dataset, split=args.train_split, mode='train', **data_kwargs)
testset = get_dataset(args.dataset, split='val', mode='val', **data_kwargs)

# dataloader
kwargs = {'num_workers': args.workers, 'pin_memory': True} if args.cuda else {}
trainloader = data.DataLoader(trainset, batch_size=args.batch_size, drop_last=True, shuffle=True, **kwargs)
valloader = data.DataLoader(testset, batch_size=args.batch_size, drop_last=False, shuffle=False, **kwargs)
nclass = trainset.num_class

# model

model = get_segmentation_model(args.model, dataset=args.dataset, backbone=args.backbone,
                               root = './encoding/models/pretrain',
                               # multi_grid=args.multi_grid, multi_dilation=args.multi_dilation, os=args.os
                               )

print(model)

# optimizer using different LR
optimizer = torch.optim.SGD(model.parameters(),lr=args.lr,momentum=args.momentum, weight_decay=args.weight_decay)

# criterions
criterion = SegmentationLosses(se_loss=args.se_loss,
                                    aux=args.aux,
                                    nclass=nclass,
                                    se_weight=args.se_weight,
                                    aux_weight=args.aux_weight)

scheduler = utils.LR_Scheduler_Head(args.lr_scheduler, args.lr, args.epochs, len(trainloader))
best_pred = 0.0

# using cuda
device = torch.device("cuda:0" if args.cuda else "cpu")
if args.cuda:
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(),
              "GPUs!")  # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model, device_ids=GPUS)
model = model.to(device)



# ==================== train =====================
train_loss = 0.0
epoch = 1
model.train()
for i, (image, dep, target) in enumerate(trainloader):
    print('1 batch')
    break

scheduler(optimizer, i, epoch, best_pred)

optimizer.zero_grad()
outputs = model(image)


loss = criterion(outputs, target)
loss.backward()
optimizer.step()

train_loss += loss.item()
