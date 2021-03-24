from dense_net import DenseNet3
from dataset import RecyclingDataset
import torch.optim as optim
import sys
from resnet import ResNet, BasicBlock
from torch import manual_seed, cuda
from torch.utils.data import random_split, DataLoader
import torch
import numpy as np

from glob import glob
import os

dataset_root = 'data/prototype_5_class/'

glass_path = dataset_root + 'glass'
metal_path = dataset_root + 'metal'
misc_plastic_label = dataset_root + 'misc_plastic'
paper_label = dataset_root + 'paper'
plastic_label = dataset_root + 'plastic'


epochs = 20
test_set_portion = .05

gpu_cuda = torch.cuda.is_available()

lr = .03
weight_decay = 5e-4

batch_size = 2
num_workers = 0

num_classes = 5

###DENSENET
depth = 100

def main():
    print("Using CUDA:      {}".format(gpu_cuda))
    # model = lambda: DenseNet3(depth=depth, num_classes=num_classes)
    model = lambda: ResNet(BasicBlock, [3, 3, 3], num_classes=num_classes)
    
    optimizer = lambda m: optim.Adam(m.parameters(), lr=lr, weight_decay=weight_decay)
    lr_scheduler = lambda o: torch.optim.lr_scheduler.StepLR(o, step_size=1, gamma=0.6)
    loss_fn = torch.nn.CrossEntropyLoss()

    model = model()

    if gpu_cuda:
        model = model.cuda()

    optimizer = optimizer(model)

    lr_scheduler = lr_scheduler(optimizer)

    dataset = RecyclingDataset(glass_path, metal_path, misc_plastic_label, paper_label, plastic_label)
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
            print('logits: ', logits.shape, 'labels: ', labels.shape)
            logits = torch.nn.Softmax(dim=1)(logits)
            _, labels = labels.max(dim=1)
            loss = loss_fn.forward(logits, labels)
            '''
            for i in range(len(logits)-1):
                if ((logits[i][0] > logits[i][1]) and labels[i][0] == 1) or ((logits[i][0] < logits[i][1]) and labels[i][0] == 0):
                    num_correct += 1
            '''
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
            '''
            for i, (im, label) in enumerate(zip(logits,labels)):
                 if abs(im[0] - label[0]) < .5:
                    num_correct += 1
            '''
            print("Test: Loss: {}  |  Accuracy:  {}".format(loss, num_correct / n_test))
        lr_scheduler.step()

if __name__ == '__main__':
    main()