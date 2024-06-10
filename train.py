import argparse
import cv2
import joblib
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import time
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import warnings

from collections import OrderedDict
from datetime import datetime
from glob import glob
from PIL import Image
from lib.dataset import Dataset
from lib.metrics import *
from lib.models import get_model
from lib.preprocess import preprocess
from lib.utils import *
from skimage import measure
from skimage.io import imread
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import datasets, models, transforms
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()

    # model training hyperparameters
    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet34',
                        help='model architecture: ' +
                        ' (default: resnet34)')
    parser.add_argument('--epochs', default=30, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--batch_size', default=32, type=int,
                        metavar='N', help='mini-batch size (default: 32)')
    parser.add_argument('--img_size', default=288, type=int,
                        help='input image size (default: 288)')
    parser.add_argument('--input_size', default=256, type=int,
                        help='input image size (default: 256)')
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--min_lr', default=1e-5, type=float,
                        help='minimum learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight-decay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')

    # preprocessing
    parser.add_argument('--scale_radius', default=True, type=str2bool)

    # data augmentation
    parser.add_argument('--rotate', default=True, type=str2bool)
    parser.add_argument('--rotate_min', default=-180, type=int)
    parser.add_argument('--rotate_max', default=180, type=int)
    parser.add_argument('--rescale', default=True, type=str2bool)
    parser.add_argument('--rescale_min', default=0.8889, type=float)
    parser.add_argument('--rescale_max', default=1.0, type=float)
    parser.add_argument('--shear', default=True, type=str2bool)
    parser.add_argument('--shear_min', default=-36, type=int)
    parser.add_argument('--shear_max', default=36, type=int)
    parser.add_argument('--translate', default=False, type=str2bool)
    parser.add_argument('--translate_min', default=0, type=float)
    parser.add_argument('--translate_max', default=0, type=float)
    parser.add_argument('--flip', default=True, type=str2bool)
    parser.add_argument('--contrast', default=True, type=str2bool)
    parser.add_argument('--contrast_min', default=0.9, type=float)
    parser.add_argument('--contrast_max', default=1.1, type=float)
    parser.add_argument('--random_erase', default=False, type=str2bool)
    parser.add_argument('--random_erase_prob', default=0.5, type=float)
    parser.add_argument('--random_erase_sl', default=0.02, type=float)
    parser.add_argument('--random_erase_sh', default=0.4, type=float)
    parser.add_argument('--random_erase_r', default=0.3, type=float)

    # dataset
    parser.add_argument('--n_splits', default=5, type=int)

    args = parser.parse_args()

    return args


def train(train_loader, model, criterion, optimizer, epoch):
    losses = AverageMeter()
    scores = AverageMeter()

    # training mode
    model.train()

    for i, (input, target) in tqdm_notebook(enumerate(train_loader), total=len(train_loader)):
        input = input.cuda()
        target = target.cuda()

        # predicts and calculates loss
        preds = model(input)
        loss = criterion(preds.view(-1), target.float())

        # updata weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # calculate score
        preds = preds_to_classes(preds)
        score = quadratic_weighted_kappa(preds, target)

        # record training process and results
        losses.update(loss.item(), input.size(0))
        scores.update(score, input.size(0))

    return losses.avg, scores.avg


def validate(val_loader, model, criterion):
    losses = AverageMeter()
    scores = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (input, target) in tqdm_notebook(enumerate(val_loader), total=len(val_loader)):
            input = input.cuda()
            target = target.cuda()

            # predicts and calculates loss
            preds = model(input)
            loss = criterion(preds.view(-1), target.float())

            # record validate process and results
            preds = preds_to_classes(preds)
            score = quadratic_weighted_kappa(preds, target)

            losses.update(loss.item(), input.size(0))
            scores.update(score, input.size(0))

    return losses.avg, scores.avg


def main():
    #----------------------------- args ---------------------------------------
    args = parse_args()

    if args.name is None:
        args.name = '%s_%s' % (args.arch, datetime.now().strftime('%m%d%H'))

    if not os.path.exists('models/%s' % args.name):
        os.makedirs('models/%s' % args.name)

    print('Config -----')
    for arg in vars(args):
        print('- %s: %s' % (arg, getattr(args, arg)))
    print('------------')

    with open('models/%s/args.txt' % args.name, 'w') as f:
        for arg in vars(args):
            print('- %s: %s' % (arg, getattr(args, arg)), file=f)

    joblib.dump(args, 'models/%s/args.pkl' % args.name)
    #--------------------------- args end -------------------------------------

    criterion = nn.MSELoss().cuda()
    num_outputs = 1
    cudnn.benchmark = True

    #--------------------------- transform ------------------------------------
    train_transform = []
    train_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.RandomAffine(
            degrees=(args.rotate_min, args.rotate_max) if args.rotate else 0,
            translate=(args.translate_min, args.translate_max) if args.translate else None,
            scale=(args.rescale_min, args.rescale_max) if args.rescale else None,
            shear=(args.shear_min, args.shear_max) if args.shear else None,
        ),
        transforms.CenterCrop(args.input_size),
        transforms.RandomHorizontalFlip(p=0.5 if args.flip else 0),
        transforms.RandomVerticalFlip(p=0.5 if args.flip else 0),
        transforms.ColorJitter(
            brightness=0,
            contrast=args.contrast,
            saturation=0,
            hue=0),
        RandomErase(
            prob=args.random_erase_prob if args.random_erase else 0,
            sl=args.random_erase_sl,
            sh=args.random_erase_sh,
            r=args.random_erase_r),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    #------------------------- transform end ----------------------------------

    #--------------------------- Load data ------------------------------------
    diabetic_retinopathy_dir = preprocess(
        'diabetic_retinopathy',
        args.img_size,
        scale=args.scale_radius)
    diabetic_retinopathy_df = pd.read_csv('inputs/diabetic-retinopathy-resized/trainLabels.csv')
    diabetic_retinopathy_img_paths = \
        diabetic_retinopathy_dir + '/' + diabetic_retinopathy_df['image'].values + '.jpeg'
    diabetic_retinopathy_labels = diabetic_retinopathy_df['level'].values

    aptos2019_dir = preprocess(
        'aptos2019',
        args.img_size,
        scale=args.scale_radius)
    aptos2019_df = pd.read_csv('inputs/train.csv')
    aptos2019_img_paths = aptos2019_dir + '/' + aptos2019_df['id_code'].values + '.png'
    aptos2019_labels = aptos2019_df['diagnosis'].values

    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=41)
    img_paths = []
    labels = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(aptos2019_img_paths, aptos2019_labels)):
        img_paths.append((np.hstack((aptos2019_img_paths[train_idx], diabetic_retinopathy_img_paths)), aptos2019_img_paths[val_idx]))
        labels.append((np.hstack((aptos2019_labels[train_idx], diabetic_retinopathy_labels)), aptos2019_labels[val_idx]))
    #------------------------- Load data end ----------------------------------

    
    # start to train models
    folds = []
    best_losses = []
    best_scores = []

    for fold, ((train_img_paths, val_img_paths), (train_labels, val_labels)) in enumerate(zip(img_paths, labels)):
        print('Fold [%d/%d]' %(fold+1, len(img_paths)))

        if os.path.exists('models/%s/model_%d.pth' % (args.name, fold+1)):
            log = pd.read_csv('models/%s/log_%d.csv' %(args.name, fold+1))
            best_loss, best_score = log.loc[log['val_loss'].values.argmin(), ['val_loss', 'val_score']].values
            folds.append(str(fold + 1))
            best_losses.append(best_loss)
            best_scores.append(best_score)
            continue

        # train
        train_set = Dataset(
            train_img_paths,
            train_labels,
            transform=train_transform)

        _, class_sample_counts = np.unique(train_labels, return_counts=True)
        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
            sampler=None)

        val_set = Dataset(
            val_img_paths,
            val_labels,
            transform=val_transform)
        val_loader = torch.utils.data.DataLoader(
            val_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4)

        # create model
        model = get_model(model_name=args.arch,
                          num_outputs=num_outputs,
                          freeze_bn=args.freeze_bn,
                          dropout_p=args.dropout_p)
        model = model.cuda()

        # set optimizer and schedular
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
                                  momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)

        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.min_lr)

        log = pd.DataFrame(index=[], columns=[
            'epoch', 'loss', 'score', 'val_loss', 'val_score'
        ])
        log = {
            'epoch': [],
            'loss': [],
            'score': [],
            'val_loss': [],
            'val_score': [],
        }

        best_loss = float('inf')
        best_score = 0
        for epoch in range(args.epochs):
            print('Epoch [%d/%d]' % (epoch + 1, args.epochs))

            # train one epoch
            train_loss, train_score = train(
                args, train_loader, model, criterion, optimizer, epoch)
            # evaluate
            val_loss, val_score = validate(args, val_loader, model, criterion)

            # update schedule
            scheduler.step()

            print('loss %.4f - score %.4f - val_loss %.4f - val_score %.4f'
                  % (train_loss, train_score, val_loss, val_score))

            # record training process
            log['epoch'].append(epoch)
            log['loss'].append(train_loss)
            log['score'].append(train_score)
            log['val_loss'].append(val_loss)
            log['val_score'].append(val_score)

            pd.DataFrame(log).to_csv('models/%s/log_%d.csv' % (args.name, fold+1), index=False)

            if val_loss < best_loss:
                torch.save(model.state_dict(), 'models/%s/model_%d.pth' % (args.name, fold+1))
                best_loss = val_loss
                best_score = val_score
                print("=> saved best model")

        print('val_loss:  %f' % best_loss)
        print('val_score: %f' % best_score)

        folds.append(str(fold + 1))
        best_losses.append(best_loss)
        best_scores.append(best_score)

        results = pd.DataFrame({
            'fold': folds + ['mean'],
            'best_loss': best_losses + [np.mean(best_losses)],
            'best_score': best_scores + [np.mean(best_scores)],
        })

        print(results)
        results.to_csv('models/%s/results.csv' % args.name, index=False)

        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
