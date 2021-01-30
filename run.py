import os
import time
import argparse
import shutil
import numpy as np
import logging

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data

import models
import datasets

def build_model(arch, pre_trained, num_seg):
    if arch == "rgb_resneXt3D64f101_bert10_FRMB":
        model = models.rgb_resneXt3D64f101_bert10_FRMB(modelPath=pre_trained, num_classes=226, length=num_seg)

    if torch.cuda.device_count() > 1:
        model=torch.nn.DataParallel(model)
    model = model.cuda()
    
    return model

def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def train(train_loader, model, criterion, criterion2, optimizer, epoch):
    batch_time = AverageMeter()
    lossesClassification = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()

    model.train()
    optimizer.zero_grad()
    
    end = time.time()

    loss_mini_batch_classification = 0.0
    acc_mini_batch = 0.0
    acc_mini_batch_top3 = 0.0
    totalSamplePerIter=0

    for i, (inputs, targets) in enumerate(train_loader):
        inputs=inputs.view(-1,length,3,input_size,input_size).transpose(1,2)
        inputs = inputs.cuda()
        targets = targets.cuda()

        output, input_vectors, sequenceOut, maskSample = model(inputs)

        prec1, prec3 = accuracy(output.data, targets, topk=(1, 3))
        acc_mini_batch += prec1.item()
        acc_mini_batch_top3 += prec3.item()

        lossClassification = criterion(output, targets)
        lossClassification = lossClassification / args.iter_size

        totalLoss=lossClassification
        loss_mini_batch_classification += lossClassification.data.item()
        totalLoss.backward()
        totalSamplePerIter +=  output.size(0)

        if (i+1) % args.iter_size == 0:
            optimizer.step()
            optimizer.zero_grad()

            lossesClassification.update(loss_mini_batch_classification, totalSamplePerIter)
            top1.update(acc_mini_batch/args.iter_size, totalSamplePerIter)
            top3.update(acc_mini_batch_top3/args.iter_size, totalSamplePerIter)
            batch_time.update(time.time() - end)

            end = time.time()

            loss_mini_batch_classification = 0
            acc_mini_batch = 0
            acc_mini_batch_top3 = 0.0
            totalSamplePerIter = 0.0

        if (i+1) % args.print_freq == 0:
            print('[%d] time: %.3f loss: %.4f' %(i,batch_time.avg,lossesClassification.avg))
            logging.info('[%d] time: %.3f loss: %.4f' %(i,batch_time.avg,lossesClassification.avg))
        
    text = '[Train] Epoch: {epoch} Prec@1 {top1.avg:.3f} Prec@3 {top3.avg:.3f} Classification Loss {lossClassification.avg:.4f}\n'
          .format(epoch = epoch, top1=top1, top3=top3, lossClassification=lossesClassification)
    print(text)
    logging.info(text)

def validate(val_loader, model, criterion, criterion2):
    batch_time = AverageMeter()
    lossesClassification = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()

    model.eval()

    end = time.time()

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(val_loader):
            inputs=inputs.view(-1,length,3,input_size,input_size).transpose(1,2)
            
            inputs = inputs.cuda()
            targets = targets.cuda()

            output, input_vectors, sequenceOut, _ = model(inputs)

            lossClassification = criterion(output, targets)
            prec1, prec3 = accuracy(output.data, targets, topk=(1, 3))

            lossesClassification.update(lossClassification.data.item(), output.size(0))
            top1.update(prec1.item(), output.size(0))
            top3.update(prec3.item(), output.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

    text = '[Eval] Prec@1 {top1.avg:.3f} Prec@3 {top3.avg:.3f} Classification Loss {lossClassification.avg:.4f}\n' 
            .format(top1=top1, top3=top3, lossClassification=lossesClassification)
    print(text)
    logging.info(text)

    return top1.avg, top3.avg, lossesClassification.avg

def save_checkpoint(state, is_best, filename, resume_path):
    cur_path = os.path.join(resume_path, filename)
    torch.save(state, cur_path)
    best_path = os.path.join(resume_path, 'model_best.pth.tar')
    if is_best:
        shutil.copyfile(cur_path, best_path)

def main(args):
    global best_prec1, best_loss

    input_size = int(224 * args.scale)
    width = int(340 * args.scale)
    height = int(256 * args.scale)

    if not os.path.exists(args.savelocation):
        os.makedirs(args.savelocation)
    
    model = build_model(args.arch, args.pre, args.num_seg)
    optimizer = AdamW(model.parameters(), lr= args.lr, weight_decay=args.weight_decay)

    criterion = nn.CrossEntropyLoss().cuda()
    criterion2 = nn.MSELoss().cuda()

    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=5, verbose=True)

    # if args.dataset=='sign':
    #     dataset="/data/AUTSL/train_img_c"
    # elif args.dataset=="signd":
    #     dataset="/data/AUTSL/train_img_c"
    # elif args.dataset=="customd":
    #     dataset="/data/AUTSL/train_img_c"
    # else:
    #     print("no dataset")
    #     return 0

    cudnn.benchmark = True
    length = 64

    scale_ratios = [1.0, 0.875, 0.75, 0.66]
    clip_mean = [114.7748, 107.7354, 99.4750] * args.num_seg * length
    clip_std = [1, 1, 1] * args.num_seg * length
    
    normalize = video_transforms.Normalize(mean=clip_mean, std=clip_std)

    train_transform = video_transforms.Compose([
        video_transforms.MultiScaleCrop((input_size, input_size), scale_ratios),
        video_transforms.RandomHorizontalFlip(),
        video_transforms.ToTensor2(),
        normalize,
    ])
    
    val_transform = video_transforms.Compose([
        video_transforms.CenterCrop((input_size)),
        video_transforms.ToTensor2(),
        normalize,
    ])

    train_file = os.path.join(args.datasetpath, args.trainlist)
    val_file = os.path.join(args.datasetpath, args.vallist)
    
    if not os.path.exists(train_file) or not os.path.exists(val_file):
        print("No split file exists in %s directory. Preprocess the dataset first" % (args.datasetpath))
    
    train_dataset = datasets.__dict__[args.dataset](root=args.datasetpath,
                                                    source=train_file,
                                                    phase="train",
                                                    modality="rgb",
                                                    is_color=True,
                                                    new_length=length,
                                                    new_width=width,
                                                    new_height=height,
                                                    video_transform=train_transform,
                                                    num_segments=args.num_seg)
    
    val_dataset = datasets.__dict__[args.dataset](root=args.datasetpath,
                                                  source=val_file,
                                                  phase="val",
                                                  modality="rgb",
                                                  is_color=True,
                                                  new_length=length,
                                                  new_width=width,
                                                  new_height=height,
                                                  video_transform=val_transform,
                                                  num_segments=args.num_seg)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    
    best_prec1 = 0
    for epoch in range(0, args.epochs):
        train(train_loader, model, criterion, criterion2, optimizer, epoch)

        if (epoch + 1) % args.save_freq == 0:
            prec1, prec3, lossClassification = validate(val_loader, model, criterion, criterion2)
            scheduler.step(lossClassification)

            if prec1 >= best_prec1:
                is_best = True
                best_prec1 = prec1
            
            checkpoint_name = "%03d_%s" % (epoch + 1, "checkpoint.pth.tar")
            text = "save checkpoint {}".format(checkpoint_name)
            print(text)
            logging.info(text)
            save_checkpoint({
                "epoch": epoch + 1,
                "arch": args.arch,
                "state_dict": model.state_dict(),
                "prec1": prec1,
                "optimizer": optimizer.state_dict()
            }, is_best, checkpoint_name, args.saveLocation)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='PyTorch Two-Stream Action Recognition')

    parser.add_argument('--dataset', '-d', default='sign',
                        choices=["sign", "signd"],
                        help='dataset: sign | signd')
    parser.add_argument('--datasetpath', default='/data/AUTSL/train_img_c',
                        help='path to datset')
    parser.add_argument('--trainlist', default='train_rgb_split01.txt',
                        help='path to train datset list')
    parser.add_argument('--vallist', default='val_rgb_split01.txt',
                        help='path to val datset list')                        

    parser.add_argument('--arch', '-a', default='rgb_resneXt3D64f101_bert10_FRMB',
                        help='models')
    parser.add_argument('--pre', default='/data/AUTSL/weights/resnet-101-64f-kinetics.pth',
                        help='models')

    parser.add_argument('-j', '--workers', default=2, type=int,
                        help='number of data loading workers (default: 2)')

    parser.add_argument('--epochs', default=200, type=int,
                        help='number of total epochs to run')
    parser.add_argument('--warmup', default=5, type=int,
                        help='warm up epoch (default: 5)')
    parser.add_argument('-b', '--batch-size', default=8, type=int,
                        help='mini-batch size (default: 8)')
    parser.add_argument('--iter-size', default=16, type=int,
                        help='iter size to reduce memory usage (default: 16)')
    parser.add_argument('--lr', '--learning-rate', default=1e-5, type=float,
                        help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', '--wd', default=1e-3, type=float,
                        help='weight decay (default: 1e-3)')
    parser.add_argument('--num-seg', default=1, type=int,
                        help='Number of segments in dataloader (default: 1)')
    parser.add_argument('--scale', default=0.5, type=float,
                        help='scale (default: 0.5)')


    parser.add_argument('--print-freq', default=100, type=int,
                        help='print frequency (default: 400)')
    parser.add_argument('--save-freq', default=10, type=int,
                        help='save frequency (default: 1)')

    parser.add_argument('--resume', default=None, type=str,
                    help='path to latest checkpoint (default: None)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')

    parser.add_argument('--savelocation', default="/data/AUTSL/checkpoint/", type=str,
                    help='path to saved checkpoint (default: /data/AUTSL/checkpoint/)')
    
    args = parser.parse_args()

    main(args)