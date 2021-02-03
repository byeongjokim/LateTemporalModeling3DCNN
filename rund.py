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
from torch.optim import lr_scheduler

import models
import datasets
import video_transforms
import swats
from opt.AdamW import AdamW

def build_model(arch, pre_trained, num_seg, resume):
    if arch == "rgb_Depth_r2plus1d_64f_34_bert10":
        model = models.rgb_Depth_r2plus1d_64f_34_bert10(num_classes=226, length=num_seg, modelPath=pre_trained)

    if resume:
        params = torch.load(resume)
        model.load_state_dict(params["state_dict"])

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

def save_checkpoint(state, is_best, filename, resume_path):
    cur_path = os.path.join(resume_path, filename)
    torch.save(state, cur_path)
    best_path = os.path.join(resume_path, 'model_best.pth.tar')
    if is_best:
        shutil.copyfile(cur_path, best_path)

def train(length, input_size, train_loader, model, criterion, criterion2, optimizer, epoch):
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

    for i, (rgb_inputs, d_inputs, targets) in enumerate(train_loader):
        rgb_inputs = rgb_inputs.view(-1, length, 3, input_size, input_size).transpose(1,2)
        rgb_inputs = rgb_inputs.cuda()

        d_inputs = d_inputs.view(-1, length, 3, input_size, input_size).transpose(1,2)
        d_inputs = d_inputs.cuda()

        targets = targets.cuda()

        output, input_vectors, sequenceOut, maskSample = model(rgb_inputs, d_inputs)

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
        
    text = '[Train] Epoch: {epoch} Prec@1 {top1.avg:.3f} Prec@3 {top3.avg:.3f} Classification Loss {lossClassification.avg:.4f}\n'.format(epoch = epoch, top1=top1, top3=top3, lossClassification=lossesClassification)
    print(text)
    logging.info(text)

def validate(length, input_size, val_loader, model, criterion, criterion2):
    batch_time = AverageMeter()
    lossesClassification = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()

    model.eval()

    end = time.time()

    with torch.no_grad():
        for i, (rgb_inputs, d_inputs, targets) in enumerate(val_loader):
            rgb_inputs = rgb_inputs.view(-1, length, 3, input_size, input_size).transpose(1,2)
            rgb_inputs = rgb_inputs.cuda()

            d_inputs = d_inputs.view(-1, length, 3, input_size, input_size).transpose(1,2)
            d_inputs = d_inputs.cuda()

            targets = targets.cuda()

            output, input_vectors, sequenceOut, _ = model(rgb_inputs, d_inputs)

            lossClassification = criterion(output, targets)
            prec1, prec3 = accuracy(output.data, targets, topk=(1, 3))

            lossesClassification.update(lossClassification.data.item(), output.size(0))
            top1.update(prec1.item(), output.size(0))
            top3.update(prec3.item(), output.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

    text = '[Eval] Prec@1 {top1.avg:.3f} Prec@3 {top3.avg:.3f} Classification Loss {lossClassification.avg:.4f}\n'.format(top1=top1, top3=top3, lossClassification=lossesClassification)
    print(text)
    logging.info(text)

    return top1.avg, top3.avg, lossesClassification.avg

# def test(length, input_size, test_loader, model, output_file):
#     output = open(output_file, "w")

#     model.eval()

#     total_pred = []

#     with torch.no_grad():
#         for i, (inputs, _) in enumerate(test_loader):
#             inputs = inputs.view(-1, length, 3, input_size, input_size).transpose(1,2)
#             inputs = inputs.cuda()

#             output, input_vectors, sequenceOut, _ = model(inputs)
            
#             _, pred = output.data.topk(1, 1, True, True)
#             total_pred += pred.tolist()
    
#     output.write(total_pred)
#     output.close()

def main(args):
    global best_prec1, best_loss

    input_size = int(224 * args.scale)
    width = int(340 * args.scale)
    height = int(256 * args.scale)

    if not os.path.exists(args.savelocation):
        os.makedirs(args.savelocation)
    
    now = time.time()
    savelocation = os.path.join(args.savelocation, str(now))
    os.makedirs(savelocation)

    logging.basicConfig(filename=os.path.join(savelocation, "log.log"), level=logging.INFO)
    
    model = build_model(args.arch, args.pre, args.num_seg, args.resume)
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

    clip_mean = [0.43216, 0.394666, 0.37645] * args.num_seg * length
    clip_std = [0.22803, 0.22145, 0.216989] * args.num_seg * length

    normalize = video_transforms.Normalize(mean=clip_mean, std=clip_std)

    train_transform = video_transforms.Compose([
        video_transforms.CenterCrop(input_size),
        video_transforms.ToTensor2(),
        normalize,
    ])
    
    val_transform = video_transforms.Compose([
        video_transforms.CenterCrop((input_size)),
        video_transforms.ToTensor2(),
        normalize,
    ])

    # test_transform = video_transforms.Compose([
    #     video_transforms.CenterCrop((input_size)),
    #     video_transforms.ToTensor2(),
    #     normalize,
    # ])
    # test_file = os.path.join(args.datasetpath, args.testlist)

    
    if not os.path.exists(args.trainlist) or not os.path.exists(args.vallist):
        print("No split file exists in %s directory. Preprocess the dataset first" % (args.datasetpath))
    
    train_dataset = datasets.__dict__[args.dataset](root=args.datasetpath,
                                                    source=args.trainlist,
                                                    phase="train",
                                                    modality="rgb",
                                                    is_color=True,
                                                    new_length=length,
                                                    new_width=width,
                                                    new_height=height,
                                                    video_transform=train_transform,
                                                    num_segments=args.num_seg)
    
    val_dataset = datasets.__dict__[args.dataset](root=args.datasetpath,
                                                  source=args.vallist,
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
        train(length, input_size, train_loader, model, criterion, criterion2, optimizer, epoch)

        if (epoch + 1) % args.save_freq == 0:
            is_best = False
            prec1, prec3, lossClassification = validate(length, input_size, val_loader, model, criterion, criterion2)
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
            }, is_best, checkpoint_name, savelocation)

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

    parser.add_argument('--dataset', '-d', default='signd',
                        choices=["sign", "signd"],
                        help='dataset: sign | signd')
    parser.add_argument('--datasetpath', default='/data/AUTSL/train_img_c',
                        help='path to datset')
    parser.add_argument('--trainlist', default='train_rgb_split01.txt',
                        help='path to train datset list')
    parser.add_argument('--vallist', default='val_rgb_split01.txt',
                        help='path to val datset list')               
    parser.add_argument('--testlist', default='test_rgb_split00.txt',
                        help='path to test datset list')

    parser.add_argument('--arch', '-a', default='rgb_Depth_r2plus1d_64f_34_bert10',
                        help='models')
    parser.add_argument('--pre', default='/data/AUTSL/weights/r2plus1d_34_clip32_ig65m_from_scratch-449a7af9.pth',
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