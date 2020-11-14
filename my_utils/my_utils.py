import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
import torchvision.transforms as transforms
import time
def accuracy_(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
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
def get_data_set(dataset_name,input_size = None):
    # data_trans = get_transform(dataset_name)
    # ------- parameters setting --------
    _rootdir = '../data'
    _batch_size = 256
    _num_workers = 0
    # -----------------------------------

    if dataset_name == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        trainsets = torchvision.datasets.CIFAR10(root=_rootdir, train=True,
                                                download=True, transform=transform_train)
        trainloaders = torch.utils.data.DataLoader(trainsets, batch_size=_batch_size ,
                                                  shuffle=True, num_workers=_num_workers)

        testsets = torchvision.datasets.CIFAR10(root=_rootdir, train=False,
                                               download=True, transform=transform_test)
        testloaders = torch.utils.data.DataLoader(testsets, batch_size=_batch_size,
                                                 shuffle=False, num_workers=_num_workers)
    if dataset_name == 'cifar100':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trainsets = torchvision.datasets.CIFAR100(root=_rootdir, train=True,
                                                  download=True, transform=transform_train)
        trainloaders = torch.utils.data.DataLoader(trainsets, batch_size=_batch_size,
                                                   shuffle=True, num_workers=_num_workers)

        testsets = torchvision.datasets.CIFAR100(root=_rootdir, train=False,
                                                 download=True, transform=transform_test)
        testloaders = torch.utils.data.DataLoader(testsets, batch_size=_batch_size,
                                                  shuffle=False, num_workers=_num_workers)
    return trainloaders,testloaders


def get_model(model_name):
    # can explored
    pass


def get_opt(net,opt=None,epoch = 20,learning_rate = 0.1):
    # can be explored more
    
    weight_decay = 5e-4
    hyparameters = {'max_epoch': epoch}

    #optimizer = optim.Adam(net.parameters(), weight_decay=weight_decay,lr=learning_rate)
    optimizer = optim.SGD(net.parameters(), lr=learning_rate,momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 100, 120])

    return optimizer, lr_scheduler, criterion, hyparameters


def train(net,train_dataset,optimizer,criterion,device,epoch):
    _total_samples = len(train_dataset.dataset)
    net.train()
    top1 = AverageMeter()
    top5 = AverageMeter()
    # top1s = AverageMeter()
    all_correct = 0
    sample = 0
    for batch_index, (images, labels) in enumerate(train_dataset):
        labels = labels.to(device)
        images = images.to(device)
        optimizer.zero_grad()
        t0 = time.time()
        outputs = net(images)
        t0 = time.time()-t0
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        correct = 0
        targets = labels
        inputs = images
        prec1, prec5 = accuracy_(outputs.data, targets.data, topk=(1, 5))
        
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))
        pred = outputs.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(labels.data.view_as(pred)).sum()
        all_correct += correct

        sample += len(outputs)
        accuracy = (100. * correct.type(torch.FloatTensor)) / len(outputs)
        aver_acc = 100. *all_correct.type(torch.FloatTensor)/(sample)
        # err1, err5 = get_error(outputs.detach(), labels, topk=(1, 5))
        # top1s.update(err1.item(), images.size(0))
        if batch_index % 10 == 0:
            print(
                'Train Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tAver_acc: {:0.6f}% lr:{}\ttop1:{:0.6f}\ttime:{:0.6f}'.format(
                    loss.item(),
                    aver_acc,
                    optimizer.param_groups[0]['lr'],
                    top1.avg,
                    t0,
                    epoch = epoch,
                    trained_samples = sample,
                    total_samples = _total_samples
                ))


def validate(net,train_dataset,optimizer,criterion,device,epoch,acc_epoch):
    # global acc_epoch
    net.eval()
    # top1 = AverageMeter()
    test_loss = 0.0  # cost function error
    correct = 0.0
    sample = 0
    for batch_index, (images, labels) in enumerate(train_dataset):
        images = images.to(device)
        labels = labels.to(device)
        t0 = time.time()
        outputs = net(images)
        t0 = time.time() - t0
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        # err1, _ = get_error(outputs.detach(), labels, topk=(1, 5))
        # top1.update(err1.item(), images.size(0))
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()
        # sample += len(outputs)
        # aver_acc = 100. * correct.type(torch.FloatTensor) / (sample)
        # print(
        #     'Training Epoch: {epoch} \tAver_acc: {:0.6f}%'.format(
        #         aver_acc,
        #         epoch=epoch,
        #     ))
    Accuracy = correct.float() / len(train_dataset.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {:.4f}\ttime:{:0.6f}'.format(
        test_loss / len(train_dataset.dataset),
        Accuracy,
        t0,
    ))
    acc_epoch.append((Accuracy, epoch))
    print()
    return Accuracy


def get_transform(name='imagenet', input_size=None,
                  scale_size=None, normalize=None, augment=True):
    pass
    # C:\Users\Admin\Desktop\experience\git_code\BinaryNet.pytorch-master\preprocess.py
    # normalize = normalize or __imagenet_stats
    # if name == 'imagenet':
    #     # scale_size = scale_size or 256
    #     # input_size = input_size or 224
    #     # if augment:
    #     #     return inception_preproccess(input_size, normalize=normalize)
    #     # else:
    #     #     return scale_crop(input_size=input_size,
    #     #                       scale_size=scale_size, normalize=normalize)
    #     pass
    # elif 'cifar' in name:
    #     input_size = input_size or 32
    #     if augment:
    #         scale_size = scale_size or 40
    #         return pad_random_crop(input_size, scale_size=scale_size,
    #                                normalize=normalize)
    #     else:
    #         scale_size = scale_size or 32
    #         return scale_crop(input_size=input_size,
    #                           scale_size=scale_size, normalize=normalize)
    # elif name == 'mnist':
    #     normalize = {'mean': [0.5], 'std': [0.5]}
    #     input_size = input_size or 28
    #     if augment:
    #         scale_size = scale_size or 32
    #         return pad_random_crop(input_size, scale_size=scale_size,
    #                                normalize=normalize)
    #     else:
    #         scale_size = scale_size or 32
    #         return scale_crop(input_size=input_size,
    #                           scale_size=scale_size, normalize=normalize)


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
        
        

def get_error(output, target, topk=(1,)):
    """Computes the error@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        # res.append(100.0 - correct_k.mul_(100.0 / batch_size))
        res.append( correct_k.mul_(100.0 / batch_size))
    return res

if __name__ == '__main__':
    pass
    # import sys
    #
    # sys.path.append("..")
    # transdata,_  = get_data_set('cifar100')
    # for i,data in enumerate(transdata):
    #     a = 1