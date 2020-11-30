from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import config as cf
import h5py
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
import json
import os
import sys
import time
import argparse
import numpy as np
import datetime
#import utils_bn
from networks import *
from torch.autograd import Variable
from metric_OOD import  eval_ood_measure

parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning_rate')
parser.add_argument('--dirsave_out', default='cifar10_0', type=str, help='where the checkpoint are save. ./checkpoint/dataset/dirsave_out')
parser.add_argument('--net_type', default='wide-resnet', type=str, help='model')
parser.add_argument('--depth', default=28, type=int, help='depth of model')
parser.add_argument('--widen_factor', default=10, type=int, help='width of model')
parser.add_argument('--dropout', default=0.3, type=float, help='dropout_rate')
parser.add_argument('--algo', default='LPBNN', type=str, help=' choose between LPBNN or BE')
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset = [cifar10/cifar100]')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--testOnly', '-t', action='store_true', help='Test mode with the saved model')
args = parser.parse_args()

# Hyper Parameter settings
use_cuda = torch.cuda.is_available()
best_acc = 0
nb_models = 1 # For better statistical results you should test on several model not just one!
ensemble_size =4
start_epoch, num_epochs, batch_size, optim_type = cf.start_epoch, cf.num_epochs, cf.batch_size, cf.optim_type

# Data Uplaod
print('\n[Phase 1] : Data Preparation')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(cf.mean[args.dataset], cf.std[args.dataset]),
]) # meanstd transformation

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cf.mean[args.dataset], cf.std[args.dataset]),
])


print("| Preparing CIFAR-10 dataset...")
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
num_classes = 10
print("| Preparing SVHN dataset...")

testset_OOD =torchvision.datasets.SVHN(root='./data', split='test', transform=transform_test, download=True)
#testset_OOD =torchvision.datasets.SVHN(root='./data', split='train', transform=transform_test, download=True)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=150, shuffle=False, num_workers=2)
testloader_OOD = torch.utils.data.DataLoader(testset_OOD, batch_size=150, shuffle=False, num_workers=2)
# Return network & file name
def getNetwork(args):
    if (args.algo == 'BE'):

        net = Wide_ResNet_BatchEnsemble(args.depth, args.widen_factor, args.dropout, num_classes,num_models=ensemble_size)
        name_algo= 'BatchEnsemble'
        print('| Building net type [wide-resnet BE]...')
    elif (args.algo == 'LPBNN'):

        net = Wide_ResNet_LPBNN(args.depth, args.widen_factor, args.dropout, num_classes, num_models=ensemble_size)
        name_algo= 'LP-BNN'
        print('| Building net type [wide-resnet LB-BNN]...')
    else:
        print('Error : Algo should be either [LPBNN / BE')
        sys.exit(0)
    file_name = 'wide-resnet-' + str(args.depth) + 'x' + str(args.widen_factor)
    return net,file_name,name_algo

# Test only option
print('\n[Test Phase] : Model setup')
assert os.path.isdir('checkpoint'), 'Error: No checkpoint directory found!'
net, file_name,name_algo = getNetwork(args)





#global_path='./checkpoint/Labia/cifar10/batch_ensemble_latent_'
#global_path='./checkpoint/Labia/cifar10/batch_ensemble_ema_latent_'
global_path=args.dirsave_out

# Model
print('\n[Phase 2] : Model setup')


if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True





## PRINT NAME FOR CHECKING
print('----------------------------------------------------')
print(' CHECK THAT "bn" IS IN THE NAME OF BATCH NORM and just that!')

# TESTING

criterion = nn.CrossEntropyLoss()






def test(model,testloader):
    model.eval()
    test_loss = 0
    correct = 0
    BS = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)
            output = model(inputs)
            if batch_idx == 0:
                output_concat = output.clone()
                target_concat = targets.clone()
            # print('output_proba',output_proba)
            else:
                output_concat = torch.cat((output_concat, output), 0)
                target_concat = torch.cat((target_concat, targets), 0)

    return output_concat, target_concat










device = 'cpu'
if use_cuda:

    device='cuda'






print(
    '----------------------------------------------------------------------------------------------------------------')
print(
    '----------------------------------------------------------------------------------------------------------------')
print(
    '----------------------------------------------------------------------------------------------------------------')
step =1

correct_list=[]


auroc_list=[]
aupr_list=[]
fpr_list=[]
ece_list=[]
for step in range(nb_models):
    print('EVALUTION of model number :',step+1)
    print('EVALUTION SVNH')

    output=torch.zeros(len(testloader_OOD.dataset),num_classes).to(device)


    print('len(testloader_OOD.dataset)',len(testloader_OOD.dataset))
    t=time.time()


    checkpoint_PATH = global_path+str(step) + os.sep + file_name + '.t7'
    print('loading ', step, ' =>', checkpoint_PATH)
    checkpoint = torch.load(checkpoint_PATH)
    net = checkpoint['net']
    net.cuda()
    output, target = test(net, testloader_OOD)


    # EVALUTION


    output_proba = output

    pred = output.argmax(dim=1, keepdim=True)

    target_onehot = torch.nn.functional.one_hot(target, num_classes)
    # one_hot = torch.cuda.FloatTensor(target.size(0), nb_class, target.size(2), target.size(3)).zero_()
    # target_onehot = one_hot.scatter_(1, target, 1)

    correct = pred.eq(target.view_as(pred)).sum().item()

    scores, _ = output_proba.max(1)
    scores = scores.view(-1)
    scores_notmnits=scores.clone()
    labels0 = target.view(-1)
    pred0 = pred.view(-1)
    #metrics_uncertainty.update(pred0*0, 10 * torch.ones_like(labels0), scores)

    #output_proba_score0 = output_proba
    output_proba_score0 = scores.clone().cpu().data.numpy()
    output_proba_score_NOTINDISTRIB = output_proba_score0


    print('time 1 : ', time.time() - t)

    _ = plt.hist(output_proba_score_NOTINDISTRIB, bins=20)
    plt.title("Histogram with 'auto' bins")

    plt.savefig('Histogram_OOD_'+name_algo+'.png')
    plt.close()

    '''testloader_NOTmnist = torch.utils.data.DataLoader(testset, batch_size=N,
                                            shuffle=False, num_workers=2)'''

    h5f = h5py.File('result_ODD_'+name_algo+'.hdf5', 'w')
    h5f.create_dataset('proba_score', data=output_proba_score_NOTINDISTRIB)

    h5f.close()



    print(
        '----------------------------------------------------------------------------------------------------------------')
    print(
        '----------------------------------------------------------------------------------------------------------------')
    print(
        '----------------------------------------------------------------------------------------------------------------')

    print('EVALUTION CIFAR10')
    print('len(testloader.dataset)',len(testloader.dataset))




    output=torch.zeros(len(testloader.dataset),num_classes).to(device)



    t=time.time()

    checkpoint_PATH = global_path+str(step) + os.sep + file_name + '.t7'
    print('loading ', step, ' =>', checkpoint_PATH)
    checkpoint = torch.load(checkpoint_PATH)
    net = checkpoint['net']
    net.cuda()
    output, target = test(net, testloader)



    # EVALUTION


    output_proba = output


    pred = output.argmax(dim=1, keepdim=True)

    target_onehot = torch.nn.functional.one_hot(target, num_classes)
    # one_hot = torch.cuda.FloatTensor(target.size(0), nb_class, target.size(2), target.size(3)).zero_()
    # target_onehot = one_hot.scatter_(1, target, 1)

    correct = pred.eq(target.view_as(pred)).sum().item()


    scores, _ = output_proba.max(1)
    scores = scores.view(-1)
    labels0 = target.view(-1)
    pred0=pred.view(-1)
    print('check label max',labels0.max())
    #|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
    conf=(torch.cat((scores_notmnits, scores), 0) )
    label=(torch.cat((10 * torch.ones_like(scores_notmnits.view(-1)).long(), labels0.long()), 0) )
    pred=(torch.cat((0 * torch.ones_like(scores_notmnits.view(-1)).long(), pred0.long()), 0) )
    res = eval_ood_measure(conf, label,pred)
    auroc, aupr, fpr, ece = res
    auroc_list.append(auroc)
    aupr_list.append(aupr)
    fpr_list.append(fpr)
    ece_list.append(ece)
    #|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
    #metrics_uncertainty.update(pred0, labels0, scores)

    #output_proba_score0=output_proba
    output_proba_score0=scores.clone().cpu().data.numpy()
    output_proba_score_INDISTRIB=output_proba_score0
    correct_list.append(correct)
    _ = plt.hist(output_proba_score_INDISTRIB, bins=20)
    plt.title("Histogram with 'auto' bins")

    plt.savefig('Histogram_ID_'+name_algo+'.png')
    plt.close()


    print('time 2 :',time.time()-t)

correct_final=np.mean(np.array(correct_list))



print('correct_list',correct_list)
print('correct = ',correct_final/ len(testloader.dataset))
print('error = ',1-correct_final/ len(testloader.dataset))



print('----------------------------------------------------------------------------------------------------------------')
print('----------------------------------------------------------------------------------------------------------------')
print('----------------------------------------------------------------------------------------------------------------')
#scores_test=metrics_uncertainty.get_scores()


'''for st in scores_test:
    print(st)
    print(scores_test[st])
    print("----------------------------------------------------------------")'''



print('----------------------------------------------------------------------------------------------------------------')


print("mean auroc = ", np.mean(np.array(auroc_list)), "mean aupr = ", np.mean(np.array(aupr_list)), " mean fpr = ", np.mean(np.array(fpr_list)),  " mean ECE = ", np.mean(np.array(ece_list)))

