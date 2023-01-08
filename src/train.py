import os
import random
import argparse

import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim
import torch.utils.data
from torch.optim.lr_scheduler import StepLR

import numpy as np
from model import LightCNN_9Layers, LightCNN_29Layers, LightCNN_29Layers_v2, SiameseModel, TripletModel, Resnet50_scratch_dag
from loss import CrossEntropy, OnlineContrastiveloss
from pytorch_metric_learning import losses, miners, distances, reducers, testers
from utils import accuracyk, MetricLog, save_checkpoint, pairselector

from sklearn.model_selection import train_test_split


parser = argparse.ArgumentParser(description='Pytorch framework for training and fine-tuning models for face recognition')

parser.add_argument("--model", default="LightCNN_29",type=str,  help='model architecture (default is LightCNN_29)')
parser.add_argument("--dataset", default="LFW", type=str, help='Dataset to be used for training(default is LFW)')
parser.add_argument("--epochs", default=50, type=int, help='epochs for training (default value is 50)')
parser.add_argument("--batch_size", default=128, type=int, help='mini-batch size for training (default value is 128)')
parser.add_argument("--learning_rate", default=0.01, type=float, help='initial learning rate for training (default value is 0.01)')
parser.add_argument("--momentum", default=0.9, type=float, help='momentum (default value is 0.9)')
parser.add_argument("--weight_decay", default=1e-4, type=float, help='weight decay (default value is 1e-4)')
parser.add_argument("--arch", default="LightCNN_29", type=str, help='model architecture (default is LightCNN_29)')
parser.add_argument("--num_classes", default= 10,type=int, help='number of classes (default value is 10)')
parser.add_argument("--save_path", default="", type=str, help='path to save the checkpoint file(default is None)')
parser.add_argument("--val_list", default="", type=str, help='path to validation list(default is None)')
parser.add_argument("--train_list", default="", type=str, help='path to training list(default is None)')




def train_epoch(model, train_loader, criterion, optimizer, lr_scheduler, device):
    train_epoch_losses = MetricLog()
    for idx, data in enumerate(train_loader):
        inputs, labels = data
        inputs = inputs.to(device, non_blocking = True)
        labels = labels.to(device, non_blocking = True)
        model = model.to(device)
        optimizer.zero_grad()
        preds_tuple = model(inputs)
        loss = criterion(preds_tuple[1], labels)
        train_epoch_losses.update(loss.item())
        loss.backward()
        optimizer.step()
    print('\nTraining set: Average loss: {}\n'.format(train_epoch_losses.avg), flush=True)
    return train_epoch_losses.avg




def val_epoch(model, val_loader, criterion, optimizer, device):
    val_losses = MetricLog()
    model.eval()
    with torch.no_grad():
        for idx, data in enumerate(val_loader):
            inputs,labels = data
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            model = model.to(device)
            preds_tuple = model(inputs)
            loss = criterion(preds_tuple[1], labels)
            val_losses.update(loss.item())
    print('\nValidation set: Average loss: {}\n'.format(val_losses.avg), flush=True)
    return val_losses.avg


def train_model(model, train_loader, val_loader, criterion, optimizer, lr_scheduler, epochs, save_path, arch):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for epoch in range(epochs):
        lr_scheduler.step()
        print('Epoch:', epoch,'LR:', lr_scheduler.get_lr())
        train_epoch(model, train_loader, criterion, optimizer, lr_scheduler, device)
        val_loss = val_epoch(model, val_loader, criterion, optimizer, device)
        save_name = save_path + 'lightCNN_' + str(epoch+1) + '_checkpoint.pth.tar'
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': arch,
            'state_dict': model.state_dict(),
            'loss': val_loss,
        }, save_name)


def train_val_split(dataset, val_size = 0.3):
    train_idx, val_idx = train_test_split(list(range(len(dataset.targets))), test_size = val_size, stratify = dataset.targets)
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(dataset, val_idx)
    return train_dataset, val_dataset

def main():
    global args
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True
    if args.model == "LightCNN_29":
        model = LightCNN_29Layers(num_classes=args.num_classes)
    elif args.model == "LightCNN_29v2":
        model = LightCNN_29Layers_v2(num_classes=args.num_classes)
    elif args.model == "LightCNN_9":
        model = LightCNN_9Layers(num_classes=args.num_classes)
    elif args.model == "SiameseModel":
        backbone = LightCNN_29Layers(num_classes=2)
        model = SiameseModel(backbone)
    elif args.model == "VGGFace2":
        model = Resnet50_scratch_dag()
    elif args.model == "ArcFace":
        pass
    else:
        print('Incorrect value for model type \n', flush=True)

    
    if args.model in ["LightCNN_29", "LightCNN_29v2", "LightCNN_9"]:
        train_transforms = transforms.Compose([transforms.RandomCrop(128), transforms.RandomHorizontalFlip(), transforms.Grayscale(num_output_channels=1), transforms.ToTensor()])
        val_transforms = transforms.Compose([transforms.RandomCrop(128), transforms.RandomHorizontalFlip(), transforms.Grayscale(num_output_channels=1), transforms.ToTensor()])
        report_cmc_path = '../results/cmc_lightcnn_plot.pdf'
    elif args.model == "VGGFace2":
        train_transforms = transforms.Compose([transforms.CenterCrop(128), transforms.Resize(256), transforms.RandomCrop(224), transforms.RdomGrayscale(p=0.2), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((131.0912/255, 103.8827/255, 91.4953/255), (1/255, 1/255, 1/255))])
        val_transforms = transforms.Compose([transforms.CenterCrop(128), transforms.Resize(256), transforms.CenterCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((131.0912/255, 103.8827/255, 91.4953/255), (1/255, 1/255, 1/255))])
        report_cmc_path = '../results/cmc_vgg_plot.pdf'

    dataset = torchvision.datasets.ImageFolder('../data/lfw-deepfunneled_processed/Train/', transform=train_transforms)
    train_dataset, val_dataset = train_val_split(dataset, val_size = 0.3)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=0, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, num_workers=0, pin_memory=True)

    criterion = losses.ContrastiveLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = args.learning_rate, weight_decay=args.weight_decay, momentum=args.momentum)
    lr_scheduler = StepLR(optimizer, step_size=20, gamma=0.1)

    train_model(model, train_loader, val_loader, criterion, optimizer, lr_scheduler, args.epochs, args.save_path, args.arch)

if __name__ == '__main__':
    main()
