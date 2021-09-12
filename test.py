'''
define the convolutinal gaussian blur
define the softmax loss

'''
import math
import time
from tqdm import tqdm
import os
import json
import yaml
import argparse
import torch
import torch.nn as nn
from torch.utils import data
import numpy as np
import pdb
from models import ModelBuilder, SegmentationModule
from lib.nn import user_scattered_collate, patch_replication_callback
from torch.autograd import Variable
import segtransforms
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import os.path as osp
import matplotlib.pyplot as plt
from skimage import io
from tensorboardX import SummaryWriter
from utils.utils import create_logger, AverageMeter, robust_binary_crossentropy, bugged_cls_bal_bce, log_cls_bal
from utils.utils import save_checkpoint as save_best_checkpoint
from utils import transforms_seg
from torchvision import transforms
from dataset.gta5_dataset import GTA5DataSet
from dataset.cityscapes_dataset import cityscapesDataSet, fake_cityscapesDataSet
from PIL import Image
from tensorboardX import SummaryWriter
import logging
IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument('--config', type=str, default='cfgs/ssda.yaml')
    return parser.parse_args()


args = get_arguments()

def mkdirs(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)

def label_mapping(input, mapping):
    output = np.copy(input)
    for ind in range(len(mapping)):
        output[input == mapping[ind][0]] = mapping[ind][1]
    return np.array(output, dtype=np.int64)

def entropy_map(v):
    return -torch.mul(v, torch.log2(v + 1e-30))

def fast_hist(a, b, n):
    k = (a >= 0) & (a < n) & (b >= 0) & (b < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)

def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))


def main():
    """Create the model and start the training."""
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    for k, v in config['common'].items():
        setattr(args, k, v)
    mkdirs(osp.join("logs/"+args.exp_name))

    logger = create_logger('global_logger', "logs/" + args.exp_name + '/log.txt')
    logger.info('{}'.format(args))
##############################

    for key, val in vars(args).items():
        logger.info("{:16} {}".format(key, val))
    logger.info("random_scale {}".format(args.random_scale))
    logger.info("is_training {}".format(args.is_training))

    h, w = map(int, args.input_size.split(','))

    h, w = map(int, args.input_size_target.split(','))
    input_size_target = (h, w)
    print(type(input_size_target[1]))
    cudnn.enabled = True
    args.snapshot_dir = args.snapshot_dir + args.exp_name
    tb_logger = SummaryWriter("logs/"+args.exp_name)
##############################

#validation data
    h, w = map(int, args.input_size_test.split(','))
    input_size_test = (h,w)
    h, w = map(int, args.com_size.split(','))
    com_size = (h, w)
    h, w = map(int, args.input_size_crop.split(','))
    h,w = map(int, args.input_size_target_crop.split(','))


    test_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])
    test_transform = transforms.Compose([
                         transforms.Resize((input_size_test[1], input_size_test[0])),
                         transforms.ToTensor(),
                         test_normalize])

    data_list_target_test = './dataset/cityscapes_list/test.txt'
    val500loader = data.DataLoader(cityscapesDataSet(
                                       args.data_dir_target,
                                       args.data_list_target_val,
                                       crop_size=input_size_test,
                                       set='val',
                                       transform=test_transform,),num_workers=args.num_workers,
                                 batch_size=1, shuffle=False, pin_memory=True)


    with open('./dataset/cityscapes_list/info.json', 'r') as fp:
        info = json.load(fp)
    mapping = np.array(info['label2train'], dtype=np.int)
    label_path_list_val = args.label_path_list_val
    gt_imgs_val = open(label_path_list_val, 'r').read().splitlines()
    gt_imgs_val = [osp.join(args.data_dir_target_val, x) for x in gt_imgs_val]

    name_classes = np.array(info['label'], dtype=np.str)
    palette= np.array(info['palette'])
    interp_val = nn.Upsample(size=(com_size[1], com_size[0]),mode='bilinear', align_corners=True)

    ####
    #build model
    ####
    builder = ModelBuilder()
    net_encoder = builder.build_encoder(
        arch=args.arch_encoder,
        fc_dim=args.fc_dim,
        weights="snapshots/" + args.exp_name + "/encoder_i_itermodel_best.pth.tar")
    net_decoder = builder.build_decoder(
        arch=args.arch_decoder,
        fc_dim=args.fc_dim,
        num_class=args.num_classes,
        weights="snapshots/" + args.exp_name + "/decoder_i_itermodel_best.pth.tar",
        use_aux=True)


    model = SegmentationModule(
        net_encoder, net_decoder, args.use_aux)

    if args.num_gpus > 1:
        model = torch.nn.DataParallel(model)
        patch_replication_callback(model)
    model.cuda()

    nets = (net_encoder, net_decoder, None, None)
    cudnn.enabled=True
    cudnn.benchmark=True
    # model.train()

    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]


    source_normalize = transforms_seg.Normalize(mean=mean,
                                                std=std)

    mean_mapping = [0.485, 0.456, 0.406]
    mean_mapping = [item * 255 for item in mean_mapping]

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)


    
    model.eval()

    val_time = time.time()
    hist = np.zeros((19,19))
    f = open(args.result_dir, 'a')
    for index, batch in tqdm(enumerate(val500loader)):
        with torch.no_grad():
            image,name = batch
            T_result = model(Variable(image).cuda(), None)
            T_result = T_result[0]
            T_result=interp_val(F.softmax(T_result, dim=1))
            conf_tea, pseudo_label = torch.max(T_result, dim=1)

            pseudo_label = pseudo_label.cpu().data[0].numpy()
            label = np.array(Image.open(gt_imgs_val[index]))
            label = label_mapping(label, mapping)
            hist += fast_hist(label.flatten(), pseudo_label.flatten(), 19)



    mIoUs = per_class_iu(hist)
    for ind_class in range(args.num_classes):
        logger.info('===>' + name_classes[ind_class] + ':\t' + str(round(mIoUs[ind_class] * 100, 2)))

    mIoUs = round(np.nanmean(mIoUs) *100, 2)

    logger.info("current mIoU {}".format(mIoUs))
    
    tb_logger.add_scalar('val mIoU', mIoUs, index)


if __name__ == '__main__':
    main()
