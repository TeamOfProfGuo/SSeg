###########################################################################
# Created by: Hang Zhang
# Email: zhang.hang@rutgers.edu
# Copyright (c) 2017
###########################################################################

import os
import copy
import yaml
import logging
import argparse
import numpy as np
from tqdm import tqdm
from addict import Dict

import torch
from torch.utils import data
from tensorboardX import SummaryWriter
import torchvision.transforms as transform
from torch.nn.parallel.scatter_gather import gather

import encoding.utils as utils
from encoding.nn import SegmentationLosses, SyncBatchNorm
from encoding.parallel import DataParallelModel, DataParallelCriterion
from encoding.datasets import get_dataset
from encoding.models import get_segmentation_model
CONFIG_PATH = 'results/psp_resnet50/config.yaml'

def get_arguments():
    '''
    parse all the arguments from command line inteface
    return a list of parsed arguments
    '''

    parser = argparse.ArgumentParser(description='semantic segmentation using PASCAL VOC')
    parser.add_argument('--config_path', type=str, help='path of a config file')
    return parser.parse_args(['--config_path', CONFIG_PATH])
    #return parser.parse_args()


class Trainer():
    def __init__(self, args):
        self.args = args
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
        self.trainloader = data.DataLoader(trainset, batch_size=args.batch_size, drop_last=True, shuffle=True, **kwargs)
        self.valloader = data.DataLoader(testset, batch_size=args.batch_size, drop_last=False, shuffle=False, **kwargs)
        self.nclass = trainset.num_class

        # model
        model = get_segmentation_model(args.model, dataset=args.dataset,
                                       backbone=args.backbone, aux=args.aux,
                                       se_loss=args.se_loss, # norm_layer=SyncBatchNorm,
                                       base_size=args.base_size, crop_size=args.crop_size,
                                       # multi_grid=args.multi_grid, multi_dilation=args.multi_dilation, os=args.os
                                       )
        print(model)
        # optimizer using different LR
        params_list = [{'params': model.pretrained.parameters(), 'lr': args.lr}, ]
        if hasattr(model, 'head'):
            params_list.append({'params': model.head.parameters(), 'lr': args.lr * 10})
        if hasattr(model, 'auxlayer'):
            params_list.append({'params': model.auxlayer.parameters(), 'lr': args.lr * 10})
        optimizer = torch.optim.SGD(params_list, lr=args.lr,
                                    momentum=args.momentum, weight_decay=args.weight_decay)
        # criterions
        self.criterion = SegmentationLosses(se_loss=args.se_loss,
                                            aux=args.aux,
                                            nclass=self.nclass,
                                            se_weight=args.se_weight,
                                            aux_weight=args.aux_weight)
        self.model, self.optimizer = model, optimizer

        # using cuda
        if args.cuda:
            self.model = DataParallelModel(self.model).cuda()
            self.criterion = DataParallelCriterion(self.criterion).cuda()

        # for writing summary
        self.writer = SummaryWriter('./results/deeplab_resnet50')
        # resuming checkpoint
        if args.resume is not None and args.resume != 'None':
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            if args.cuda:
                self.model.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
            if not args.ft:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        # clear start epoch if fine-tuning
        if args.ft:
            args.start_epoch = 0

        # lr scheduler
        self.scheduler = utils.LR_Scheduler_Head(args.lr_scheduler, args.lr, args.epochs, len(self.trainloader))
        self.best_pred = 0.0

    def training(self, epoch):
        train_loss = 0.0
        self.model.train()
        for i, (image, dep, target) in enumerate(self.trainloader):
            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()
            outputs = self.model(image)
            loss = self.criterion(*outputs, target)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            if (i+1) % 50 == 0:
                print('epoch {}, step {}, loss {}'.format(epoch + 1, i + 1, train_loss / 50))
                self.writer.add_scalar('train_loss', train_loss / 50, epoch * len(self.trainloader) + i)
                train_loss = 0.0

    def train_n_evaluate(self):

        best_val_loss = 0.0
        for epoch in range(self.args.epochs):
            # run on one epoch
            print("\n===============train epoch {}/{} ==========================\n".format(epoch + 1, self.args.epochs))

            # one full pass over the train set
            self.training(epoch)

            # evaluate for one epoch on the validation set
            print('\n===============start testing, training epoch {}\n'.format(epoch))
            pixAcc, mIOU, loss = self.validation(epoch)
            print('evaluation pixel acc {}, mean IOU {}, loss {}'.format(pixAcc, mIOU, loss))

            # save the best model
            is_best = False
            new_pred = (pixAcc + mIOU) / 2
            if new_pred > self.best_pred:
                is_best = True
                self.best_pred = new_pred
            utils.save_checkpoint({'epoch': epoch + 1,
                                   'state_dict': self.model.module.state_dict(),
                                   'optimizer': self.optimizer.state_dict(),
                                   'best_pred': self.best_pred}, self.args, is_best)

    def validation(self, epoch):
        # Fast test during the training
        def eval_batch(model, image, target):
            outputs = model(image)
            #outputs = gather(outputs, 0, dim=0)  # check this line for parallel computing
            pred = outputs[0]
            loss = self.criterion(pred, target)
            target = target.cuda()
            correct, labeled = utils.batch_pix_accuracy(pred.data, target)
            inter, union = utils.batch_intersection_union(pred.data, target, self.nclass)

            return correct, labeled, inter, union, loss

        self.model.eval()
        total_inter, total_union, total_correct, total_label, total_loss = 0, 0, 0, 0, 0
        for i, (image, dep, target) in enumerate(self.valloader):
            with torch.no_grad():
                correct, labeled, inter, union, loss = eval_batch(self.model, image, target)

            total_correct += correct
            total_label += labeled
            total_inter += inter
            total_union += union
            total_loss += loss.item()
            pixAcc = 1.0 * total_correct / (np.spacing(1) + total_label)
            IOU = 1.0 * total_inter / (np.spacing(1) + total_union)
            mIOU = IOU.mean()

            if i % 10 == 0:
                print('eval mean IOU {}'.format(mIOU))
            loss = total_loss / len(self.valloader)

            self.writer.add_scalar("mean_iou/val", mIOU, epoch)
            self.writer.add_scalar("pixel accuracy/val", pixAcc, epoch)

        return pixAcc, mIOU, loss


if __name__ == "__main__":
    config = get_arguments()
    # configuration
    args = Dict(yaml.safe_load(open(config.config_path)))
    args.cuda = (args.use_cuda and torch.cuda.is_available())
    torch.manual_seed(args.seed)

    trainer = Trainer(args)
    print('Starting Epoch:', trainer.args.start_epoch)
    print('Total Epoches:', trainer.args.epochs)
    trainer.train_n_evaluate()


