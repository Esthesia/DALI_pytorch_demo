import argparse
import os
import shutil
import time
import random

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models

from dataloader import Dataset

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset with subdirectories named "train" and "val)')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                    ' | '.join(model_names) +
                    ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('-v', '--val-batch-size', default=500, type=int,
                    metavar='N', help='mini-batch size for val - must evenly divide number of examples (default: 500)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--fp16', action='store_true',
                    help='Run model fp16 mode.')
parser.add_argument('--use-dali', action='store_true',
                    help='Use DALI loader instead of torchvision.')
parser.add_argument('--dali-cpu', action='store_true',
                    help='Runs CPU based version of DALI pipeline.')
parser.add_argument('--static-loss-scale', type=float, default=1,
                    help='Static loss scale, positive power of 2 values can improve fp16 convergence.')
parser.add_argument('--dynamic-loss-scale', action='store_true',
                    help='Use dynamic loss scaling.  If supplied, this argument supersedes ' +
                    '--static-loss-scale.')
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument("--rand_factor",default=[2,2],action='store_true',
                    help='RandAugment factor input [M, N]')

cudnn.benchmark = True

args = parser.parse_args()

if not len(args.data):
    raise Exception("error: too few arguments")

args.distributed = False
if 'WORLD_SIZE' in os.environ:
    args.distributed = int(os.environ['WORLD_SIZE']) > 1

# make apex optional
# if args.fp16 or args.distributed:
#     try:
#         from apex.parallel import DistributedDataParallel as DDP
#         from apex.fp16_utils import *
#     except ImportError:
#         raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")

# item() is a recent addition, so this helps with backward compatibility.
def to_python_float(t):
    if hasattr(t, 'item'):
        return t.item()
    else:
        return t[0]

def main():
    global args

    args.gpu = 0
    args.world_size = 1

    # if args.distributed:
    #     args.gpu = args.local_rank % torch.cuda.device_count()
    #     torch.cuda.set_device(args.gpu)
    #     torch.distributed.init_process_group(backend='nccl',
    #                                          init_method='env://')
    #     args.world_size = torch.distributed.get_world_size()

    args.total_batch_size = args.world_size * args.batch_size

    if args.fp16:
        assert torch.backends.cudnn.enabled, "fp16 mode requires cudnn backend to be enabled."

    if args.static_loss_scale != 1.0:
        if not args.fp16:
            print("Warning:  if --fp16 is not used, static_loss_scale will be ignored.")

    # create model
    print("=> creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch]()

    model = model.cuda()
    
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    # if args.fp16:
    #     model = network_to_half(model)
    # if args.distributed:
    #     # shared param/delay all reduce turns off bucketing in DDP, for lower latency runs this can improve perf
    #     # for the older version of APEX please use shared_param, for newer one it is delay_allreduce
    #     model = DDP(model, delay_allreduce=True)

    # if args.fp16:
    #     optimizer = FP16_Optimizer(optimizer,
    #                                static_loss_scale=args.static_loss_scale,
    #                                dynamic_loss_scale=args.dynamic_loss_scale)

    print(f"{args.use_dali} on using_DALI option and {args.dali_cpu} on using CPU with DALI")
    
    # Create dataloader
    dataset = Dataset(data_dir=args.data,
                      batch_size=args.batch_size,
                      val_batch_size=args.val_batch_size,
                      workers=args.workers,
                      world_size=args.world_size,
                      use_dali=args.use_dali,
                      dali_cpu=args.dali_cpu,
                      fp16=args.fp16,
                      rand_factor=args.rand_factor,
                      )

    if args.evaluate:
        validate(dataset, model, criterion)
        return

    total_time = AverageMeter()
    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch

        avg_train_time = train(dataset, model, criterion, optimizer, epoch)
        total_time.update(avg_train_time)

        # Set dataset to GPU mode, to speed up validation - it's pretty slow otherwise with this new resizing
        # dataset.prep_for_val()

        # evaluate on validation set
        # [prec1, prec5] = validate(dataset, model, criterion)
        # [prec1, prec5] = [0, 0]
        # remember best prec@1 and save checkpoint
        if args.local_rank == 0:
            # is_best = prec1 > best_prec1
            # best_prec1 = max(prec1, best_prec1)
            # save_checkpoint({
            #     'epoch': epoch + 1,
            #     'arch': args.arch,
            #     'state_dict': model.state_dict(),
            #     'best_prec1': best_prec1,
            #     'optimizer': optimizer.state_dict(),
            # }, is_best)
            ####################################
            if epoch == args.epochs - 1:
                print("DONE")
                # print('##Perf  {0}'.format(args.total_batch_size / total_time.avg))

        # reset DALI iterators
        dataset.reset()


def augment_list():

    l = [
        # (AutoContrast, 0, 1),
        # (Equalize, 0, 1),
        ("Invert", 0, 1),
        ("Rotate", 0, 30),
        # (Posterize, 0, 4),
        # (Solarize, 0, 256),
        # (SolarizeAdd, 0, 110), 
        # (Color, 0.1, 1.9),
        ("Contrast", 0.1, 1.9),
        ("Brightness", 0.1, 1.9),
        # ("Sharpness", 0.1, 1.9),
        ("ShearX", 0., 0.3),
        ("ShearY", 0., 0.3),
        # (CutoutAbs, 0, 40),
        ("TranslateXabs", 0., 100),
        ("TranslateYabs", 0., 100),
    ]

    return l


def train(dataset, model, criterion, optimizer, epoch, warmup_batches=10, prof_batches=100):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    data_arr = []
    
    train_loader = dataset.get_train_loader()

    # switch to train mode
    model.train()
    end = time.time()
    end_time = torch.cuda.Event(enable_timing=True)
    end_time.record()

    aug_list = augment_list()
    checker = 0
    
    for i, data in enumerate(train_loader):
        input = data[0]
        target = data[1]
        # print(target)
        # ops_list = []
        # for i in range(len(input)):
        #     ops = random.choices(aug_list,k=2)
        #     ops_list.append(ops)
        # print(f"{len(ops_list)} with per ops_list is {len(ops_list[0])}")
        check_time = torch.cuda.Event(enable_timing=True)
        check_time.record()

        # measure data loading time
        data_time.update(time.time() - end)
        print(f"data time is {time.time()-end}")
        
        adjust_learning_rate(optimizer, epoch, i, len(train_loader))
        
        # # compute output
        # output = model(input)
        # loss = criterion(output, target)

        # # measure accuracy and record loss
        # prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

        # if args.distributed:
        #     reduced_loss = reduce_tensor(loss.data)
        #     prec1 = reduce_tensor(prec1)
        #     prec5 = reduce_tensor(prec5)
        # else:
        #     reduced_loss = loss.data

        # losses.update(to_python_float(reduced_loss), input.size(0))
        # top1.update(to_python_float(prec1), input.size(0))
        # top5.update(to_python_float(prec5), input.size(0))

        # # compute gradient and do SGD step
        # optimizer.zero_grad()
        # if args.fp16:
        #     optimizer.backward(loss)
        # else:
        #     loss.backward()
        # optimizer.step()

        # torch.cuda.synchronize()
        # measure elapsed time
        
        batch_time.update(time.time() - end)
        
        data_arr.append((end_time, check_time))
        
        end_time = torch.cuda.Event(enable_timing=True)
        end_time.record()
        end = time.time()
        ####################################
        # if args.local_rank == 0 and i % args.print_freq == 0 and i > 1:
        # print('Epoch: [{0}][{1}/{2}]\t'
        #         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        #         'Speed {3:.3f} ({4:.3f})\t'
        #         'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'.format(
        #         epoch, i, len(train_loader),
        #         args.total_batch_size / batch_time.val,
        #         args.total_batch_size / batch_time.avg,
        #         batch_time=batch_time.avg,
        #         data_time=data_time.val))
        print(f"{epoch} : {batch_time.val} {data_time.val}")
        checker = checker + 1
        if checker == 5:
            break
    


    sum_time = 0
    for i in range(len(data_arr)):
        data_arr[i][0].synchronize()
        datatime = data_arr[i][0].elapsed_time(data_arr[i][1])/1000
        sum_time += datatime
        print(f"{i} iterations {datatime}")
        
    print(f"{sum_time} at corresponding epochs basd on CUDA events")
    
        
    return batch_time.avg


def validate(dataset, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    val_loader = dataset.get_val_loader()

    # switch to evaluate mode
    model.eval()

    end = time.time()

    for i, data in enumerate(val_loader):
        input = data[0]
        target = data[1]

        # compute output
        with torch.no_grad():
            output = model(input)
            loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

        if args.distributed:
            reduced_loss = reduce_tensor(loss.data)
            prec1 = reduce_tensor(prec1)
            prec5 = reduce_tensor(prec5)
        else:
            reduced_loss = loss.data

        losses.update(to_python_float(reduced_loss), input.size(0))
        top1.update(to_python_float(prec1), input.size(0))
        top5.update(to_python_float(prec5), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.local_rank == 0 and i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Speed {2:.3f} ({3:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader),
                   args.total_batch_size / batch_time.val,
                   args.total_batch_size / batch_time.avg,
                   batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return [top1.avg, top5.avg]


# def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
#     torch.save(state, filename)
#     if is_best:
#         shutil.copyfile(filename, 'model_best.pth.tar')


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


def adjust_learning_rate(optimizer, epoch, step, len_epoch):
    """LR schedule that should yield 76% converged accuracy with batch size 256"""
    factor = epoch // 30

    if epoch >= 80:
        factor = factor + 1

    lr = args.lr * (0.1 ** factor)

    """Warmup"""
    if epoch < 5:
        lr = lr * float(1 + step + epoch * len_epoch) / (5. * len_epoch)

    if(args.local_rank == 0 and step % args.print_freq == 0 and step > 1):
        print("Epoch = {}, step = {}, lr = {}".format(epoch, step, lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= args.world_size
    return rt

if __name__ == '__main__':
    main()
