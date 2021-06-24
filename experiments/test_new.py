###########################################################################
# Created by: Hang Zhang
# Email: zhang.hang@rutgers.edu
# Copyright (c) 2017
###########################################################################

import os
import argparse
import numpy as np
from tqdm import tqdm

import torch
from torch.utils import data
import torchvision.transforms as transform
from torch.nn.parallel.scatter_gather import gather

import encoding.utils as utils
from encoding.nn import SegmentationLosses, SyncBatchNorm
from encoding.parallel import DataParallelModel, DataParallelCriterion
from encoding.datasets import get_dataset, test_batchify_fn
from encoding.models import get_model, get_segmentation_model, MultiEvalModule
#from model_mapping import rename_weight_for_head


# colour map
label_colours = [(0, 0, 0),  # 0=background
                 # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                 (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                 # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                 (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
                 # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
                 (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
                 # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
                 (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)]

class Options():
    def __init__(self):
        parser = argparse.ArgumentParser(description='PyTorch Segmentation')
        # model and dataset
        parser.add_argument('--model', type=str, default='deeplab', help='model name (default: encnet)')
        parser.add_argument('--backbone', type=str, default='resnet50', help='backbone name (default: resnet50)')
        parser.add_argument('--dataset', type=str, default='pascal_aug', help='dataset name (default: pascal12)')
        parser.add_argument('--workers', type=int, default=4, metavar='N', help='dataloader threads')
        parser.add_argument('--base-size', type=int, default=520, help='base image size')
        parser.add_argument('--crop-size', type=int, default=480, help='crop image size')
        parser.add_argument('--train-split', type=str, default='train', help='dataset train split (default: train)')

        # training hyper params
        parser.add_argument('--aux', action='store_true', default= False, help='Auxilary Loss')
        parser.add_argument('--se-loss', action='store_true', default= False, help='Semantic Encoding Loss SE-loss')
        parser.add_argument('--se-weight', type=float, default=0.2, help='SE-loss weight (default: 0.2)')
        parser.add_argument('--batch-size', type=int, default=2, metavar='N', help='input batch size for training')
        parser.add_argument('--test-batch-size', type=int, default=2, metavar='N', help='batch size for testing')

        # cuda, seed and logging
        parser.add_argument('--no-cuda', action='store_true', default= False, help='disables CUDA training')
        parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

        # checking point
        parser.add_argument('--model_path', type=str, default='./runs/pascal_voc/deeplab/resnet50/default/checkpoint.pth.tar',
                            help='put the path to resuming file if needed')

        # evaluation option
        parser.add_argument('--eval', action='store_true', default= True, help='evaluating mIoU')
        parser.add_argument('--test-val', action='store_true', default=False, help='generate masks on val set')
        parser.add_argument('--export', type=str, default=None, help='put the path to resuming file if needed')
        parser.add_argument('--acc-bn', action='store_true', default= False, help='Re-accumulate BN statistics')

        # multi grid dilation option
        parser.add_argument("--multi-grid", action="store_true", default=False, help="use multi grid dilation policy")
        parser.add_argument('--multi-dilation', nargs='+', type=int, default=None, help="multi grid dilation list")
        parser.add_argument('--os', type=int, default=8, help='output stride default:8')
        parser.add_argument('--no-deepstem', action="store_true", default=False, help='backbone without deepstem')
        # the parser
        self.parser = parser

    def parse(self):
        args = self.parser.parse_args()
        args.cuda = not args.no_cuda and torch.cuda.is_available()
        print(args)
        return args


def main():

    args = Options().parse()
    torch.manual_seed(args.seed)
    # args.test_batch_size = torch.cuda.device_count()
    args.test_batch_size = 1

    # data transforms
    input_transform = transform.Compose([transform.ToTensor(),
                                         transform.Normalize([.485, .456, .406], [.229, .224, .225])])
    # dataset
    if args.eval:
        testset = get_dataset(args.dataset, split='val', mode='val', transform=input_transform)
    elif args.test_val:
        testset = get_dataset(args.dataset, split='val', mode='val', transform=input_transform)
    else:
        testset = get_dataset(args.dataset, split='test', mode='test', transform=input_transform)   # use val as test data
    # dataloader
    loader_kwargs = {'num_workers': args.workers, 'pin_memory': True} if args.cuda else {}
    testloader = data.DataLoader(testset, batch_size=args.test_batch_size,
                                drop_last=False, shuffle=False,
                                #collate_fn=test_batchify_fn,
                                **loader_kwargs)
    # model
    model = get_segmentation_model(args.model, dataset=args.dataset,
                                   backbone=args.backbone, aux=args.aux, se_loss=args.se_loss,
                                   norm_layer=torch.nn.BatchNorm2d, # if args.acc_bn else SyncBatchNorm,
                                   base_size=args.base_size, crop_size=args.crop_size,
                                   #multi_grid=args.multi_grid, multi_dilation=args.multi_dilation, os=args.os,
                                   #no_deepstem=args.no_deepstem
                                   )

    # load model params
    if args.model_path is not None and os.path.isfile(args.model_path):
        print("=> loading checkpoint '{}'".format(args.model_path))
        model.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu'))['state_dict'])
    else:
        raise RuntimeError("=> no checkpoint found")
    print(model)

    #Re - accumulate BN statistics
    if args.acc_bn:
        from encoding.utils.precise_bn import update_bn_stats
        data_kwargs = {'transform': input_transform, 'base_size': args.base_size, 'crop_size': args.crop_size}
        trainset = get_dataset(args.dataset, split=args.train_split, mode='train', **data_kwargs)
        trainloader = data.DataLoader(ReturnFirstClosure(trainset), batch_size=args.batch_size,
                                      drop_last=True, shuffle=True, **loader_kwargs)
        print('Reseting BN statistics')
        # model.apply(reset_bn_statistics)
        model.cuda()
        update_bn_stats(model, trainloader)

    if args.export:
        torch.save(model.state_dict(), args.export + '.pth')
        return

    # evaluation metrics
    metrics = utils.SegmentationMetric(testset.num_class)

    validate(args, model, testloader, metrics)


def validate(args, model, testset, testloader, metrics):
    """Do validation and return specified samples"""

    # scales = [1.0] if args.dataset == 'citys' else scales = [0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25]
    scales = [1.0]
    evaluator = MultiEvalModule(model, testset.num_class, scales=scales)
    evaluator.eval()

    #dir for output mask
    out_dir= './results/deeplab_resnet50/out'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    tbar = tqdm(testloader)
    for i, (image, dst) in enumerate(tbar):   # in test mode dst is a tuple
        if args.eval:
            with torch.no_grad():
                #predicts = evaluator.parallel_forward(image)
                predicts = evaluator.forward(image)     # [B, 21, h, w] with h=w=480
                metrics.update(dst, predicts)
                pixAcc, mIoU = metrics.get()
                tbar.set_description('pixAcc: %.4f, mIoU: %.4f' % (pixAcc, mIoU))
        else:
            with torch.no_grad():
                #outputs = evaluator.parallel_forward(image)
                predict = evaluator.forward(image)
                pred = testset.make_pred(torch.max(predict, 1)[1].cpu().numpy())
                impath = dst[0]
                mask = utils.get_mask_pallete(pred, args.dataset)
                outname = os.path.splitext(impath)[0] + '.png'
                mask.save(os.path.join(out_dir, outname))

    if args.eval:
        print( 'pixAcc: %.4f, mIoU: %.4f' % (pixAcc, mIoU))


class ReturnFirstClosure(object):
    def __init__(self, data):
        self._data = data

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        outputs = self._data[idx]
        return outputs[0]


def decode_labels(mask, num_images=1, num_classes=21):
  """Decode batch of segmentation masks.

  Args:
    mask: result of inference after taking argmax.
    num_images: number of images to decode from the batch.
    num_classes: number of classes to predict (including background).

  Returns:
    A batch with num_images RGB images of the same size as the input.
  """
  n, h, w, c = mask.shape
  assert (n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' \
                            % (n, num_images)
  outputs = np.zeros((num_images, h, w, 3), dtype=np.uint8)
  for i in range(num_images):
    img = Image.new('RGB', (len(mask[i, 0]), len(mask[i])))
    pixels = img.load()
    for j_, j in enumerate(mask[i, :, :, 0]):
      for k_, k in enumerate(j):
        if k < num_classes:
          pixels[k_, j_] = label_colours[k]
    outputs[i] = np.array(img)
  return outputs

if __name__ == "__main__":
    main()