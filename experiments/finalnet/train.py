###########################################################################
# Model Sample - Team of Prof. Guo
# Created by: Hammond Liu
# Copyright (c) 2021
###########################################################################

import os
import sys
import time
import yaml
import numpy as np
from addict import Dict

import torch
import torch.nn as nn
from torch.utils import data
from tensorboardX import SummaryWriter
import torchvision.transforms as transform

# Default Work Dir: /scratch/[NetID]/SSeg/experiments/[Model]/
BASE_DIR = os.path.join(os.getcwd(), '../..')
sys.path.append(BASE_DIR)
# Path for config and summary files
TRAIN_PATH = './config/train.yaml'
# CONFIG_PATH = './config/%s.yaml' % sys.argv[1]
SMY_PATH = os.path.join('./results/', sys.argv[1])
# GPU ids (only when there are multiple GPUs)
GPUS = [0, 1]

import net.utils as utils
from net.nn import SegmentationLoss
from net.datasets import get_dataset
from net.models import get_segmentation_model
from config.config import get_config

class Trainer():
    def __init__(self, args):
        self.args = args

        # data transforms
        input_transform = transform.Compose([
            transform.ToTensor(),  # convert RGB [0,255] to FloatTensor in range [0, 1]
            transform.Normalize([.485, .456, .406], [.229, .224, .225])])  # mean and std based on imageNet
        if args.dataset in ('nyud', 'nyud_tmp'):
            dep_transform = transform.Compose([
                transform.ToTensor(),
                transform.Normalize(mean=[0.2798], std=[0.1387])  # mean and std for depth
            ])
        elif args.dataset == 'sunrgbd':
            dep_transform = transform.Compose([
                transform.ToTensor(),
                transform.Lambda(lambda x: x.to(torch.float)),
                transform.Normalize(mean=[19025.15], std=[9880.92])  # mean and std for depth
            ])
        else:
            raise ValueError('Unable to transform depth on the selected dataset.')

        # dataset
        data_kwargs = {'transform': input_transform, 'dep_transform': dep_transform,
                       'base_size': args.base_size, 'crop_size': args.crop_size}
        trainset = get_dataset(args.dataset, root=sys.argv[2], split=args.train_split, mode='train', **data_kwargs)
        testset = get_dataset(args.dataset, root=sys.argv[2], split='val', mode='val', **data_kwargs)
        # dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True} if args.cuda else {}
        self.trainloader = data.DataLoader(trainset, batch_size=args.batch_size, drop_last=True, shuffle=True, **kwargs)
        self.valloader = data.DataLoader(testset, batch_size=args.batch_size, drop_last=False, shuffle=False, **kwargs)
        self.nclass = trainset.num_class

        # config
        # config = Dict(yaml.safe_load(open(CONFIG_PATH)))
        config = get_config(sys.argv[1])
        for k, v in config.items():
            print('[%s]: %s' % (k, v))

        # model and params
        model = get_segmentation_model(args.model, dataset=args.dataset, backbone=args.backbone, 
                                       pretrained=True, root='../../encoding/models/pretrain',
                                       config=config)
        print(model)

        self.config = model.config
        # for k, v in self.config.items():
        #     print('[%s]: %s' % (k, v))

        # optimizer using different LR
        base_ids = list(map(id, model.rgb_base.parameters())) + list(map(id, model.dep_base.parameters()))
        other_params = filter(lambda p: id(p) not in base_ids, model.parameters())
        self.optimizer = torch.optim.SGD([{'params': model.rgb_base.parameters(), 'lr': args.lr},
                                          {'params': model.dep_base.parameters(), 'lr': args.lr},
                                          {'params': other_params, 'lr': args.lr * 10}],
                                         lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

        # criterions
        self.criterion = SegmentationLoss(se_loss=args.se_loss,
                                          aux=self.config.decoder_args.aux,
                                          nclass=self.nclass,
                                          se_weight=args.se_weight,
                                          aux_weight=args.aux_weight)
        # lr scheduler
        self.scheduler = utils.LR_Scheduler_Head(args.lr_scheduler, args.lr, args.epochs, len(self.trainloader))
        self.best_pred = (0.0, 0.0)

        # using cuda
        self.device = torch.device("cuda:0" if args.cuda else "cpu")
        if args.cuda:
            if torch.cuda.device_count() > 1:
                print("Let's use", torch.cuda.device_count(), "GPUs!")  # [30,xxx]->[10,...],[10,...],[10,...] on 3 GPUs
                model = nn.DataParallel(model, device_ids=GPUS)
                self.multi_gpu = True
            else:
                self.multi_gpu = False
        self.model = model.to(self.device)

        # for writing summary
        if not os.path.isdir(SMY_PATH):
            utils.mkdir(SMY_PATH)
        self.writer = SummaryWriter(SMY_PATH)
        # image_sample = next(iter(self.trainloader))
        # self.writer.add_graph(model, (image_sample[0].to(self.device), image_sample[1].to(self.device)))

        # resuming checkpoint
        if args.resume is not None and args.resume != 'None':
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            if self.multi_gpu:
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

    def training(self, epoch):
        train_loss = 0.0
        self.model.train()

        total_inter, total_union, total_correct, total_label, total_loss = 0, 0, 0, 0, 0
        for i, (image, dep, target) in enumerate(self.trainloader):
            self.scheduler(self.optimizer, i, epoch, sum(self.best_pred))
            self.optimizer.zero_grad()
            
            if self.args.early_fusion:
                image_with_dep = torch.cat((image, dep), 1)
                image_with_dep, dep, target = image_with_dep.to(self.device), dep.to(self.device), target.to(self.device)
                outputs = self.model(image_with_dep, dep)
            else:
                image, dep, target = image.to(self.device), dep.to(self.device), target.to(self.device)
                outputs = self.model(image, dep)

            loss = self.criterion(*outputs, target)
            loss.backward()
            self.optimizer.step()

            correct, labeled = utils.batch_pix_accuracy(outputs[0].data, target)
            inter, union = utils.batch_intersection_union(outputs[0].data, target, self.nclass)
            total_correct += correct
            total_label += labeled
            total_inter += inter
            total_union += union
            train_loss += loss.item()

            if (i+1) % 50 == 0:
                print('epoch {}, step {}, loss {}'.format(epoch + 1, i + 1, train_loss / 50))
                self.writer.add_scalar('train_loss', train_loss / 50, epoch * len(self.trainloader) + i)
                train_loss = 0.0
        pixAcc = 1.0 * total_correct / (np.spacing(1) + total_label)
        IOU = 1.0 * total_inter / (np.spacing(1) + total_union)
        mIOU = IOU.mean()
        print('epoch {}, pixel Acc {}, mean IOU {}'.format(epoch + 1, pixAcc, mIOU))
        self.writer.add_scalar("mean_iou/train", mIOU, epoch)
        self.writer.add_scalar("pixel accuracy/train", pixAcc, epoch)

    def train_n_evaluate(self):

        results = Dict({'miou': [], 'pix_acc': []})

        for epoch in range(self.args.epochs):
            # run on one epoch
            print("\n===============train epoch {}/{} ==========================\n".format(epoch+1, self.args.epochs))

            # one full pass over the train set
            self.training(epoch)

            # evaluate for one epoch on the validation set
            print('\n===============start testing, training epoch {}\n'.format(epoch+1))
            pixAcc, mIOU, loss = self.validation(epoch)
            print('evaluation pixel acc {}, mean IOU {}, loss {}'.format(pixAcc, mIOU, loss))

            results.miou.append(round(mIOU, 6))
            results.pix_acc.append(round(pixAcc, 6))

            # save the best model
            is_best = False
            new_pred = (round(mIOU, 6), round(pixAcc, 6))
            if sum(new_pred) > sum(self.best_pred):
                is_best = True
                self.best_pred = new_pred
                best_state_dict = self.model.module.state_dict() if self.multi_gpu else self.model.state_dict()
            utils.save_checkpoint({'epoch': epoch + 1,
                                   'state_dict': self.model.module.state_dict() if self.multi_gpu else self.model.state_dict(),
                                   'optimizer': self.optimizer.state_dict(),
                                   'best_pred': self.best_pred}, self.args, is_best)

        final_miou, final_pix_acc = sum(results.miou[-5:]) / 5, sum(results.pix_acc[-5:]) / 5
        final_result = '\nPerformance of last 5 epochs\n[mIoU]: %4f\n[Pixel_Acc]: %4f\n[Best Pred]: %s\n' % (final_miou, final_pix_acc, self.best_pred)
        print(final_result)
        
        # Export weights if needed
        if self.args.export or self.best_pred[0] > 0.455:
            export_info = '/%s_%s_%s' % (self.args.model, self.args.dataset, int(time.time()))
            torch.save(best_state_dict, SMY_PATH + export_info + '.pth')
            with open(SMY_PATH + export_info + '.txt', 'w') as f:
                f.write(str(self.config) + '\n')
                f.write(final_result + '\n')
                f.write(str(self.model))

    def validation(self, epoch):
        # Fast test during the training
        def eval_batch(model, image, dep, target):
            # model, image, target already moved to gpus
            pred = model(image, dep)
            loss = self.criterion(*pred, target)
            correct, labeled = utils.batch_pix_accuracy(pred[0].data, target)
            inter, union = utils.batch_intersection_union(pred[0].data, target, self.nclass)
            return correct, labeled, inter, union, loss

        self.model.eval()
        total_inter, total_union, total_correct, total_label, total_loss = 0, 0, 0, 0, 0
        for i, (image, dep, target) in enumerate(self.valloader):
            # image, dep, target = image.to(self.device), dep.to(self.device), target.to(self.device)
            if self.args.early_fusion:
                image_with_dep = torch.cat((image, dep), 1)
                image_with_dep, dep, target = image_with_dep.to(self.device), dep.to(self.device), target.to(self.device)
            else:
                image, dep, target = image.to(self.device), dep.to(self.device), target.to(self.device)

            with torch.no_grad():
                correct, labeled, inter, union, loss = eval_batch(self.model, image_with_dep if self.args.early_fusion else image, dep, target)

            total_correct += correct
            total_label += labeled
            total_inter += inter
            total_union += union
            total_loss += loss.item()
            pixAcc = 1.0 * total_correct / (np.spacing(1) + total_label)
            IOU = 1.0 * total_inter / (np.spacing(1) + total_union)
            mIOU = IOU.mean()

            if i % 20 == 0:
                print('eval mean IOU {}'.format(mIOU))
            loss = total_loss / len(self.valloader)

            self.writer.add_scalar("mean_iou/val", mIOU, epoch)
            self.writer.add_scalar("pixel accuracy/val", pixAcc, epoch)

        return pixAcc, mIOU, loss


if __name__ == "__main__":
    start_time = time.time()
    print('[Exp Name]:', sys.argv[1])
    print("-------mark program start----------")
    # configuration
    args = Dict(yaml.safe_load(open(TRAIN_PATH)))
    args.cuda = (args.use_cuda and torch.cuda.is_available())
    args.resume = None if args.resume == 'None' else args.resume
    torch.manual_seed(args.seed)

    trainer = Trainer(args)
    # import pdb; pdb.set_trace()
    print('Starting Epoch:', trainer.args.start_epoch)
    print('Total Epoches:', trainer.args.epochs)
    trainer.train_n_evaluate()

    exp_time_mins = int(time.time() - start_time) // 60
    print('[Time]: %.2fh' % (exp_time_mins / 60))


