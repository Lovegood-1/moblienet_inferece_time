
#from vgg import vgg_cifar100
#net = vgg_cifar100()
from mobilenets import mobilenet
#net = mobilenet(100,False)
from mobilenets_con import mobilenet_con
net = mobilenet_con(100,False)
import torch
import torch.nn as nn
import argparse
import sys
sys.path.append("..")
import my_utils.my_utils as ut
# import allconv
# from hy_class import classifier_hy


parser = argparse.ArgumentParser()
parser.add_argument('-gpu', type=bool, default=True, help='use gpu or not')
parser.add_argument('-w', type=int, default=0, help='number of workers for dataloader')
parser.add_argument('-b', type=int, default=32, help='batch size for dataloader')
parser.add_argument('-s', type=bool, default=True, help='whether shuffle the dataset')
parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
parser.add_argument('-lr', type=float, default=0.05, help='initial learning rate')

parser.add_argument('-dataset_name', type=str, default='cifar100')
parser.add_argument('-model_name', type=str)
parser.add_argument('-opt_strate', type=str)
parser.add_argument('-epoch', type=int, default=100, help='number of epoch')
args = parser.parse_args()
# =====================================

# ========= data load
import time
import os
time_now = time.strftime("%Y%m%d-%H%M", time.localtime())
root = os.getcwd()
name_model = net._get_name() + '_baseline_'
_path =root + '/'+ name_model +time_now
if not os.path.exists(_path):
    os.makedirs(_path)


# ========= data load
train_dataset, test_dataset = ut.get_data_set(args.dataset_name)

# ========= load model
if torch.cuda.device_count() > 7:
     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
     net = nn.DataParallel(net, device_ids=[0, 2])
     net.to(device)
else:
     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
     net.to(device)

# ========= optimizer and lr
optimizer, lr_scheduler, criterion, heperpapar = ut.get_opt(net, args.opt_strate, args.epoch)
acc_epoch = []

# ========= begin loop
for epoch in range(heperpapar['max_epoch']):
    # ut.train(net, train_dataset, optimizer, criterion, device, epoch)
    acc = ut.validate(net,test_dataset,optimizer,criterion,device,epoch,acc_epoch)
    
    if lr_scheduler != None:
        lr_scheduler.step()
    if epoch %20 ==0 :
        torch.save(net, '{}/{}_dict_{}.pt'.format(_path,name_model,epoch))
# ========= save file

state = {
        'top1': acc_epoch,
    }
torch.save(state, '{}/{}acc_epoch.pth'.format(_path,name_model))