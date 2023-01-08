import argparse

import torch
import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR
import numpy as np

from model import LightCNN_9Layers, LightCNN_29Layers, LightCNN_29Layers_v2, SiameseModel, TripletModel, resnet50_scratch_dag, iresnet18, iresnet34, iresnet50, iresnet100, iresnet200
from loss import CrossEntropy, OnlineContrastiveloss

from train import train_epoch, val_epoch, train_model, train_val_split

from utils import generate_embeddings, generate_embeddings_v2, cosine_similarity_matrix, get_cmc_scores, plot_cmc_curve, pairselector

from pytorch_metric_learning import losses, miners, distances, reducers, testers


parser = argparse.ArgumentParser(description='Pytorch framework for training and fine-tuning models for face recognition')

parser.add_argument("--model", default="LightCNN_29", help='model architecture (default is LightCNN_29)')
parser.add_argument("--dataset", default="LFW", type=str, help='Dataset to be used for training(default is LFW)')
parser.add_argument("--epochs", default=50, type=int, help='epochs for training (default value is 50)')
parser.add_argument("--batch_size", default=128, type=int, help='mini-batch size for training (default value is 128)')
parser.add_argument("--learning_rate", default=0.01, type=float, help='initial learning rate for training (default value is 0.01)')
parser.add_argument("--momentum", default=0.9, type=float, help='momentum (default value is 0.9)')
parser.add_argument("--weight_decay", default=1e-4, type=float, help='weight decay (default value is 1e-4)')
parser.add_argument("--arch", default="LightCNN_29", help='model architecture (default is LightCNN_29)')
parser.add_argument("--num_classes", default= 10,type=int, help='number of classes (default value is 10)')
parser.add_argument("--save_path", default="", type=str, help='path to save the checkpoint file(default is None)')
parser.add_argument("--val_list", default="", type=str, help='path to validation list(default is None)')
parser.add_argument("--train_list", default="", type=str, help='path to training list(default is None)')


def set_parameter_requires_grad(model, feature_extracting, num_layers):
    if feature_extracting:
        num = 0
        for param in model.parameters():
            num = num + 1
            if num > len(list(model.parameters())) - num_layers:
               param.requires_grad = True
            else:
               param.requires_grad = False


def main():
    global args
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True

    if args.model == "LightCNN_29":
        model = LightCNN_29Layers(num_classes=79077)
        model = torch.nn.DataParallel(model)
        model.load_state_dict(torch.load('../models/LightCNN_29Layers_checkpoint.pth.tar')['state_dict'])
        set_parameter_requires_grad(model, feature_extracting=True, num_layers=10)
        model.module.fc2 = torch.nn.Linear(256, args.num_classes)
    elif args.model == "LightCNN_29v2":
        model = LightCNN_29Layers_v2(num_classes=80013)
        model = torch.nn.DataParallel(model)
        model.load_state_dict(torch.load('../models/LightCNN_29Layers_V2_checkpoint.pth.tar')['state_dict'])
        set_parameter_requires_grad(model, feature_extracting=True, num_layers=10)
        model.module.fc2 = torch.nn.Linear(256, args.num_classes)
    elif args.model == "LightCNN_9":
        model = LightCNN_9Layers(num_classes=79077)
        model = torch.nn.DataParallel(model)
        model.load_state_dict(torch.load('../models/LightCNN_9Layers_checkpoint.pth.tar')['state_dict'])
        set_parameter_requires_grad(model, feature_extracting=True, num_layers=10)
        model.module.fc2 = torch.nn.Linear(256, args.num_classes)
    elif args.model == "VGGFace2":
        model = resnet50_scratch_dag(weights_path = '../models/resnet50_scratch_dag.pth')
        set_parameter_requires_grad(model, feature_extracting=True, num_layers=10)
    elif args.model == "SiameseModel":
        backbone = LightCNN_29Layers(num_classes=79077)
        backbone = torch.nn.DataParallel(backbone)
        backbone.load_state_dict(torch.load('../models/LightCNN_29Layers_checkpoint.pth.tar')['state_dict'])
        set_parameter_requires_grad(backbone, feature_extracting=True, num_layers=10)
        backbone.module.fc2 = torch.nn.Linear(256, 2)
        model = SiameseModel(backbone)
    elif args.model == "ArcFace":
        model = iresnet18(pretrained=True)
        model.load_state_dict(torch.load('../models/ms1mv3_arcface_r18_fp16/backbone.pth'))
        set_parameter_requires_grad(model, feature_extracting=True, num_layers=10)
    else:
        print('Incorrect value for model type \n', flush=True)

    if args.dataset == "LFW":
        if args.model in ["LightCNN_29", "LightCNN_29v2", "LightCNN_9"]:
            train_transforms = transforms.Compose([transforms.RandomCrop(128), transforms.RandomHorizontalFlip(), transforms.Grayscale(num_output_channels=1), transforms.ToTensor()])
            val_transforms = transforms.Compose([transforms.RandomCrop(128), transforms.RandomHorizontalFlip(), transforms.Grayscale(num_output_channels=1), transforms.ToTensor()])
            report_cmc_path = '../results/cmc_lightcnn_plot.pdf'
        elif args.model == "VGGFace2":
            train_transforms = transforms.Compose([transforms.CenterCrop(128), transforms.Resize(256), transforms.RandomCrop(224), transforms.RandomGrayscale(p=0.2), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((131.0912/255, 103.8827/255, 91.4953/255), (1/255, 1/255, 1/255))])
            val_transforms = transforms.Compose([transforms.CenterCrop(128), transforms.Resize(256), transforms.CenterCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((131.0912/255, 103.8827/255, 91.4953/255), (1/255, 1/255, 1/255))])
            report_cmc_path = '../results/cmc_vgg_plot.pdf'
        elif args.model == "ArcFace":
            train_transforms = transforms.Compose([transforms.CenterCrop(112), transforms.RandomGrayscale(p=0.2), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            val_transforms = transforms.Compose([transforms.CenterCrop(112), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            report_cmc_path = '../results/cmc_arcface_plot.pdf'
        else:
            pass
        dataset = torchvision.datasets.ImageFolder('../data/lfw-deepfunneled_processed/Train/', transform=train_transforms)
    elif args.dataset == "SurvFace":
        if args.model in ["LightCNN_29", "LightCNN_29v2", "LightCNN_9"]:
            train_transforms = transforms.Compose([transforms.Resize((128,128)), transforms.RandomHorizontalFlip(), transforms.Grayscale(num_output_channels=1), transforms.ToTensor()])
            val_transforms = transforms.Compose([transforms.Resize((128,128)), transforms.Grayscale(num_output_channels=1), transforms.ToTensor()])
            report_cmc_path = '../results/cmc_lightcnn_SurvFace_plot.pdf'
        elif args.model == "VGGFace2":
            train_transforms = transforms.Compose([transforms.Resize((224,224)), transforms.RandomGrayscale(p=0.2), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((131.0912/255, 103.8827/255, 91.4953/255), (1/255, 1/255, 1/255))])
            val_transforms = transforms.Compose([transforms.Resize((224,224)), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((131.0912/255, 103.8827/255, 91.4953/255), (1/255, 1/255, 1/255))])
            report_cmc_path = '../results/cmc_vggface2_SurvFace_plot.pdf'
        elif args.model == "ArcFace":
            train_transforms = transforms.Compose([transforms.Resize((112,112)), transforms.RandomGrayscale(p=0.2), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            val_transforms = transforms.Compose([transforms.Resize((112,112)), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            report_cmc_path = '../results/cmc_arcface_SurvFace_plot.pdf'
        else:
            pass
        dataset = torchvision.datasets.ImageFolder('../data/QMUL-SurvFace/training_set/', transform=train_transforms)


    train_dataset, val_dataset = train_val_split(dataset, val_size = 0.3)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=0, pin_memory=True) # Classes may still be imbalaced during forward pass (Ref: https://github.com/adambielski/siamese-triplet) 
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, num_workers=0, pin_memory=True)

    criterion = OnlineContrastiveloss(margin = 2, pairselector=pairselector())
    criterion = losses.ContrastiveLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = args.learning_rate, weight_decay=args.weight_decay, momentum=args.momentum)
    lr_scheduler = StepLR(optimizer, step_size=20, gamma=0.1)

    train_model(model, train_loader, val_loader, criterion, optimizer, lr_scheduler, args.epochs, args.save_path, args.arch)


    if args.dataset == "LFW":
        if args.model in ["LightCNN_29", "LightCNN_29v2", "LightCNN_9"]:
            transform = transforms.Compose([transforms.ToPILImage(), transforms.CenterCrop(128), transforms.ToTensor()])
            probe_labels, probe_embeddings = np.array(generate_embeddings(model, '../data/lfw-deepfunneled_processed/Test/probe/', transform))
            gallery_labels, gallery_embeddings = np.array(generate_embeddings(model, '../data/lfw-deepfunneled_processed/Test/gallery/', transform))
        elif args.model == "VGGFace2":
            transform = transforms.Compose([transforms.ToPILImage(), transforms.CenterCrop(128), transforms.Resize(224), transforms.ToTensor(), transforms.Normalize((131.0912/255, 103.8827/255, 91.4953/255), (1/255, 1/255, 1/255))])
            probe_labels, probe_embeddings = np.array(generate_embeddings_v2(model, '../data/lfw-deepfunneled_processed/Test/probe/', transform))
            gallery_labels, gallery_embeddings = np.array(generate_embeddings_v2(model, '../data/lfw-deepfunneled_processed/Test/gallery/', transform))
        elif args.model == "ArcFace":
            transform = transforms.Compose([transforms.ToPILImage(),transforms.CenterCrop(112), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            probe_labels, probe_embeddings = np.array(generate_embeddings_v2(model, '../data/lfw-deepfunneled_processed/Test/probe/', transform))
            gallery_labels, gallery_embeddings = np.array(generate_embeddings_v2(model, '../data/lfw-deepfunneled_processed/Test/gallery/', transform))
    elif args.dataset == "SurvFace":
        if args.model in ["LightCNN_29", "LightCNN_29v2", "LightCNN_9"]:
            transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((128,128)), transforms.ToTensor()])
            probe_labels, probe_embeddings = np.array(generate_embeddings(model, '../data/QMUL-SurvFace/Face_Identification_Test_Set/mated_probe_p/', transform))
            gallery_labels, gallery_embeddings = np.array(generate_embeddings(model, '../data/QMUL-SurvFace/Face_Identification_Test_Set/gallery_p/', transform))
        elif args.model == "VGGFace2":
            transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize((131.0912/255, 103.8827/255, 91.4953/255), (1/255, 1/255, 1/255))])
            probe_labels, probe_embeddings = np.array(generate_embeddings_v2(model, '../data/QMUL-SurvFace/Face_Identification_Test_Set/mated_probe_p/', transform))
            gallery_labels, gallery_embeddings = np.array(generate_embeddings_v2(model, '../data/QMUL-SurvFace/Face_Identification_Test_Set/gallery_p/', transform))
        elif args.model == "ArcFace":
            transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((112,112)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            probe_labels, probe_embeddings = np.array(generate_embeddings_v2(model, '../data/QMUL-SurvFace/Face_Identification_Test_Set/mated_probe_p/', transform))
            gallery_labels, gallery_embeddings = np.array(generate_embeddings_v2(model, '../data/QMUL-SurvFace/Face_Identification_Test_Set/gallery_p/', transform))


    sim_matrix = cosine_similarity_matrix(probe_embeddings, gallery_embeddings)

    cmc_scores = get_cmc_scores(sim_matrix)
    print(cmc_scores, flush = True)

    plot_cmc_curve(cmc_scores, report_cmc_path, (10,10))

if __name__ == '__main__':
    main()
