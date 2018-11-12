from __future__ import print_function
import os
import warnings

warnings.simplefilter("ignore")

# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from pytorch.utils import Logger, format_time
from pytorch import models
from torch.optim.lr_scheduler import MultiStepLR
import time


def getmodel(cls=61):
    model_conv = models.resnet50(pretrained=True)
    if cls != 1000:
        num_ftrs = model_conv.fc.in_features
        model_conv.fc = nn.Linear(num_ftrs, cls)
    return model_conv


class DotDict(dict):
    def __getattr__(self, name):
        return self[name]


if __name__ == '__main__':

    args = DotDict({
        'batch_size': 32,
        'batch_mul': 4,
        'val_batch_size': 10,
        'cuda': True,
        'model': '',
        'train_plot': False,
        'epochs': [180],
        'try_no': '1_resnet50',
        'logdir': 'logs/pytorch',
        'imsize': [224],
        'imsize_l': [256],
        'traindir': '/root/palm/DATA/plant/train',
        'valdir': '/root/palm/DATA/plant/validate',
        'workers': 16,
        'resume': False,
    })
    logger = Logger(os.path.join(args.logdir, args.try_no))
    logger.text_summary('Describe', 'ResNet50', 0)
    logger.text_summary('Describe', 'Batch size: 32*4', 1)
    logger.text_summary('Describe', 'Input size: 224/256', 2)
    best_acc = 0
    best_no = 0
    start_epoch = 1
    model = getmodel(1000).cuda()

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    optimizer = torch.optim.SGD(model.parameters(), 0.1,
                                momentum=0.9,
                                weight_decay=1e-4,
                                nesterov=False,
                                )
    scheduler = MultiStepLR(optimizer, [30, 60, 90, 120, 150])
    criterion = nn.CrossEntropyLoss().cuda()
    zz = 0
    for i in range(len(args.epochs)):
        train_dataset = datasets.ImageFolder(
            args.traindir,
            transforms.Compose([
                transforms.Resize(args.imsize[i]),
                transforms.RandomResizedCrop(args.imsize[i]),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))
        trainloader = torch.utils.data.DataLoader(train_dataset,
                                                  batch_size=args.batch_size,
                                                  shuffle=True,
                                                  num_workers=args.workers,
                                                  pin_memory=False)
        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(args.valdir, transforms.Compose([
                transforms.Resize(args.imsize[i]),
                transforms.CenterCrop(args.imsize[i]),
                transforms.ToTensor(),

                normalize,
            ])),
            batch_size=args.val_batch_size,
            num_workers=args.workers,
            pin_memory=False)

        # model = torch.nn.DataParallel(model).cuda()
        cudnn.benchmark = args.batch_size > 1
        if args.resume:
            if args.resume is True:
                args['resume'] = f'pytorch/checkpoint/try_{args.try_no}best.t7'
            if os.path.isfile(args.resume):
                print("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume)
                # start_epoch = checkpoint['epoch']
                best_prec1 = checkpoint['acc']
                model.load_state_dict(checkpoint['net'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(args.resume, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(args.resume))


        def train(epoch):
            print('\nEpoch: %d/%d' % (epoch, args.epochs[i]))
            model.train()
            train_loss = 0
            correct = 0
            total = 0
            optimizer.zero_grad()
            start_time = time.time()
            last_time = start_time
            for batch_idx, (inputs, targets) in enumerate(trainloader):
                inputs, targets = inputs.to('cuda'), targets.to('cuda')
                outputs = model(inputs)
                # targets = torch.cat((targets, targets, targets, targets, targets))

                # bs, ncrops, c, h, w = inputs.size()
                # outputs = outputs.view(bs, ncrops, -1).mean(1)

                loss = criterion(outputs, targets) / args.batch_mul
                loss.backward()
                train_loss += loss.item() * args.batch_mul
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                lfs = (batch_idx + 1) % args.batch_mul
                if lfs == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                step_time = time.time() - last_time
                last_time = time.time()
                try:
                    print(f'\r{" "*(len(lss))}', end='')
                except NameError:
                    pass
                lss = f'{batch_idx}/{len(trainloader)} | ' + \
                      f'ETA: {format_time(step_time*(len(trainloader)-batch_idx))} - ' + \
                      f'loss: {train_loss/(batch_idx+1):.{3}} - ' + \
                      f'acc: {correct/total:.{5}}'
                print(f'\r{lss}', end='')

            for tag, value in model.named_parameters():
                tag = tag.replace('.', '/')
                logger.histo_summary(tag, value.data.cpu().numpy(), epoch)
                logger.histo_summary(tag + '/grad', value.grad.data.cpu().numpy(), epoch)
            logger.scalar_summary('acc', correct/total, epoch)
            logger.scalar_summary('loss', train_loss/(batch_idx+1), epoch)
            print(f'\r '
                  f'ToT: {format_time(time.time() - start_time)} - '
                  f'loss: {train_loss/(batch_idx+1):.{3}} - '
                  f'acc: {correct/total:.{5}}', end='')
            optimizer.step()
            optimizer.zero_grad()
            # scheduler2.step()


        def test(epoch):
            global best_acc, best_no
            model.eval()
            test_loss = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(val_loader):
                    inputs, targets = inputs.to('cuda'), targets.to('cuda')
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    """"""
                    # bs, ncrops, c, h, w = inputs.size()
                    # outputs = outputs.view(bs, ncrops, -1).mean(1)
                    """"""
                    test_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()

                    # progress_bar(batch_idx, len(val_loader), 'Acc: %.3f%%'
                    #              % (100. * correct / total))
            logger.scalar_summary('val_acc', correct/total, epoch)
            logger.scalar_summary('val_loss', test_loss/(batch_idx+1), epoch)
            print(f' - val_acc: {correct / total:.{5}}')

            # Save checkpoint.
            acc = 100. * correct / total
            # print('Saving..')
            state = {
                'optimizer': optimizer.state_dict(),
                'net': model.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            if acc > best_acc:
                torch.save(state, f'pytorch/checkpoint/try_{args.try_no}best.t7')
                best_acc = acc
                best_no = correct
            torch.save(state, f'pytorch/checkpoint/try_{args.try_no}temp.t7')


        for epoch in range(start_epoch, start_epoch + args.epochs[i]):
            scheduler.step()
            train(epoch)
            test(epoch)
            print(f'best: {best_acc}: {best_no}/{len(val_loader)*args.val_batch_size}')
        start_epoch += args.epochs[i]
