import os
import sys
import argparse
from datetime import datetime
import time
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
from models.resnet_56 import ResNet56
from utils.untils_svhn import get_training_dataloader, get_test_dataloader, WarmUpLR

parser = argparse.ArgumentParser()

parser.add_argument('--data_root', type=str,default='./data/svhn')
parser.add_argument('-w', type=int, default=2, help='number of workers for dataloader')
parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
parser.add_argument('-s', type=bool, default=True, help='whether shuffle the dataset')
parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--dist_prob', type=float, default=0.06)
parser.add_argument('--block_size', type=int, default=6)
parser.add_argument('--alpha', type=float, default=30)
args, unparsed = parser.parse_known_args()


def train(epoch):
    start = time.time()
    train_loss = 0.0
    correct = 0.0
    train_matrix = None

    net.train()
    for batch_index, (images, labels) in enumerate(training_loader):

        if epoch <= args.warm:
            warmup_scheduler.step()

        images = Variable(images)
        labels = Variable(labels)
        labels = labels.cuda()
        images = images.cuda()
        optimizer.zero_grad()
        outputs, covout = net(images)
        if train_matrix is None:
            train_matrix = covout
        else:
            train_matrix = torch.cat((train_matrix, covout), dim=0)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    end = time.time()

    print("-------------------------------------------------------------------------------------------")
    print(type(train_matrix), train_matrix.shape)
    print('Train set Epoch: {epoch}  Average_loss: {loss:.4f}, Accuracy: {acc:.4f}, Run_time: {Run_time:.4f}'.format(epoch=epoch,
                                                                                           loss=train_loss / len(
                                                                                               training_loader.dataset),
                                                                                           acc=correct.float() / len(
                                                                                               training_loader.dataset),
                                                                                                Run_time = end - start))

    return (train_loss / len(training_loader.dataset)), (
            correct.float() / len(training_loader.dataset)), train_matrix

def eval(train_acc):
    test_loss = 0.0
    correct = 0.0
    correct_lei = np.zeros(100)
    test_matrix = None

    net.eval()
    for (images, labels) in test_loader:
        images = Variable(images)
        labels = Variable(labels)
        images = images.cuda()
        labels = labels.cuda()
        outputs, covout = net(images)
        if test_matrix is None:
            test_matrix = covout
        else:
            test_matrix = torch.cat((test_matrix, covout), dim=0)
        loss = loss_function(outputs, labels)
        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()
        correct_lei_flag = (preds == labels).squeeze()

        for i in range(0, len(labels)):
            if correct_lei_flag[i].item():
                correct_lei[labels[i]] = correct_lei[labels[i]] + 1

    print('Test set: Average_loss: {:.4f}, Accuracy: {:.4f}, Mingzhong: {:.4f}'.format(
        test_loss / len(test_loader.dataset),
        correct.float() / len(test_loader.dataset),
        (correct.float() / len(test_loader.dataset))/train_acc
    ))

    print(type(test_matrix), test_matrix.shape)

    return (test_loss / len(test_loader.dataset)), (correct.float() / len(test_loader.dataset)), (
                correct.float() / len(test_loader.dataset)) / train_acc, correct_lei, test_matrix

def cov(train_matrix, test_matrix):
    r_shape, e_shape = train_matrix.shape, test_matrix.shape
    train_matrix, test_matrix = train_matrix.view(r_shape[0], -1), \
                                test_matrix.view(e_shape[0], -1)
    train_mean, test_mean = torch.mean(train_matrix, dim=0), torch.mean(test_matrix, dim=0)
    tct_matrix = train_matrix[r_shape[0]-e_shape[0]: r_shape[0], :]
    n_dim = train_matrix.shape[1]
    cov_abs = []
    tct_matrix = tct_matrix - train_mean
    test_matrix = test_matrix - test_mean
    for i in range(n_dim):
        rsp_matrix = tct_matrix[:, i].view(e_shape[0], 1)
        mul_mt = rsp_matrix * test_matrix
        cov_ins = torch.sum(mul_mt, dim=0) / (e_shape[0] - 1)
        abs_cov = torch.abs(cov_ins)
        cov_abs.append((torch.sum(abs_cov) / abs_cov.shape[0]).cpu().item())
    return np.sum(cov_abs) / (len(cov_abs))

if __name__ == '__main__':

    tongji = []
    for xunhuan in range(0, 5):

        training_loader = get_training_dataloader(args.data_root,
                                                  (0.5, 0.5, 0.5),
                                                  (0.5, 0.5, 0.5),
                                                  num_workers=args.w,
                                                  batch_size=args.b,
                                                  shuffle=args.s
                                                  )

        test_loader = get_test_dataloader(args.data_root,
                                                  (0.5, 0.5, 0.5),
                                                  (0.5, 0.5, 0.5),
                                                  num_workers=args.w,
                                                  batch_size=args.b,
                                                  shuffle=args.s
                                                  )


        data_for_train = []
        data_for_test = []
        correct_lei = []
        cov_list = []

        net = ResNet56(depth=56, num_classes=10, dist_prob=args.dist_prob, block_size   =args.block_size, alpha=args.alpha, nr_steps=len(training_loader) * args.epochs).cuda()

        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)
        iter_per_epoch = len(training_loader)
        warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)

        small_cov = 100.0
        best_acc = 0.0
        for epoch in range(1, args.epochs + 1):
            if epoch > args.warm:
                train_scheduler.step()

            train_loss, train_acc, train_matrix = train(epoch)
            test_loss, acc, Mingzhong, correct_lei_1, test_matrix = eval(train_acc)
            cov_start = time.time()
            cov_item = cov(train_matrix, test_matrix)
            cov_end = time.time()
            print('cov_item is: ', cov_item, 'cov cost time is ', cov_end-cov_start)
            if small_cov > cov_item:
                small_cov = cov_item
                print('small_cov:', small_cov)
            cov_list.append(cov_item)

            if best_acc < acc:
                best_acc = acc
                torch.save(net.state_dict(), "save_model/resnet56_100/resnet56" + "_" + str(time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())) + "_" + str(epoch) + ".pth")
                print('best_acc:', best_acc)

            data_for_train.append([train_loss, float(train_acc)])
            data_for_test.append([test_loss, float(acc), float(Mingzhong)])
            correct_lei.append(correct_lei_1)

        print('best_acc:', best_acc)

        Pd_data_for_covs = pd.DataFrame(cov_list)
        Pd_data_for_covs.to_csv("log/resnet56_100" + "_" + str(time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())) + ".csv",header=False, index=True)
        Pd_data_for_train = pd.DataFrame(data_for_train)
        Pd_data_for_train.to_csv("log/resnet56_100" + "_" + str(time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())) + ".csv", header=False, index=True)
        Pd_data_for_test = pd.DataFrame(data_for_test)
        Pd_data_for_test.to_csv("log/resnet56_100" + "_" + str(time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())) + ".csv", header=False, index=True)
        correct_lei_pd = pd.DataFrame(correct_lei)
        correct_lei_pd.to_csv("log/resnet56_100" + "_" + str(time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())) + ".csv", header=False, index=True)
        tongji.append([best_acc, float(best_acc)])
        xunhuan = xunhuan + 1

    tongji_pd = pd.DataFrame(tongji)
    tongji_pd.to_csv("log/resnet56_100" + "_" + str(time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())) + ".csv", header=False, index=True)

