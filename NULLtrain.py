from model import LeNet, ResNet, BasicBlock
from dataset import DrinkDataset

import torch.optim as optim
import sys

from torch import manual_seed, cuda
from torch.utils.data import random_split, DataLoader
import torch
import numpy as np

from glob import glob
import os

dataset_root = 'data/'
bottle_path = dataset_root + '0'
can_path = dataset_root + '1'


epochs = 20
test_set_portion = .2

gpu_cuda = torch.cuda.is_available()

lr = .03
weight_decay = 5e-4

batch_size = 16
num_workers = 0

out_channels = 2

def main():
    print("Using CUDA:      {}".format(gpu_cuda))
    model = lambda: ResNet(BasicBlock, [3, 3, 3], num_classes=out_channels)
    #LeNet(n_classes=out_channels)
    
    optimizer = lambda m: optim.Adam(m.parameters(), lr=lr, weight_decay=weight_decay)
    lr_scheduler = lambda o: torch.optim.lr_scheduler.StepLR(o, step_size=1, gamma=0.6)
    loss_fn = torch.nn.BCEWithLogitsLoss(weight=torch.tensor([.5,.5]))


    model = model()

    if gpu_cuda:
        model = model.cuda()

    optimizer = optimizer(model)

    lr_scheduler = lr_scheduler(optimizer)

    dataset = DrinkDataset(bottle_path, can_path)

    n_test = int(len(dataset) * test_set_portion)
    n_train = len(dataset) - n_test
    manual_seed(101)
    train_set, test_set = random_split(dataset, [n_train, n_test])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, \
        num_workers=num_workers, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, \
        num_workers=num_workers, pin_memory=torch.cuda.is_available())
    
    for epoch in range(epochs):
        model.train()

        n_batches = len(train_loader)
        num_correct = 0

        print('------- EPOCH [{} / {}] -------'.format(epoch + 1, epochs))

        #TRAIN
        for batch_idx, (imgs, labels) in enumerate(train_loader):

            if gpu_cuda:
                logits = model(imgs).cuda()
            else:
                logits = model(imgs)

            print(torch.stack([logits, labels]))
            loss = loss_fn.forward(logits, labels.float())

            for i in range(len(logits)-1):
                if ((logits[i][0] > logits[i][1]) and labels[i][0] == 1) or ((logits[i][0] < logits[i][1]) and labels[i][0] == 0):
                    num_correct += 1
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            logits.detach()

            print("Batch: {}/{}  |  Loss: {}  |  Accuracy: {}".format(batch_idx + 1, n_batches, loss, num_correct / n_train))
        
        
        #TEST
        model.eval()
        num_correct = 0
        #counter = np.zeroes(shape=5, dtype=int)
        with torch.set_grad_enabled(False):
            for batch_idx, (imgs, labels) in enumerate(test_loader):
                #imgs, labels = map(lambda x: x.to(device, dtype=torch.float32), (imgs, labels))
                if gpu_cuda:
                    logits = model(imgs).cuda()
                else:
                    logits = model(imgs)
                #loss = loss_fn(logits, labels)
                loss = loss_fn.forward(logits, labels.float())
            for i, (im, label) in enumerate(zip(logits,labels)):
                 if abs(im[0] - label[0]) < .5:
                    num_correct += 1
            print("Test: Loss: {}  |  Accuracy:  {}".format(loss, num_correct / n_test))
        lr_scheduler.step()

if __name__ == '__main__':
    main()