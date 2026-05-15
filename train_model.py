from model import my_resnet
from dataset import AnimalDataset
import torch
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
from torchvision.transforms import Compose, Resize, ToTensor


def get_args():
    parser = ArgumentParser("animal classifier")
    parser.add_argument("--batch_size", "-b", type = int, default=16)
    parser.add_argument("--epochs", "-e", type = int, default=30)
    parser.add_argument("--logging", "-l",type = str, default="./tensorboard" )
    parser.add_argument("--train_model", "-t",type = str, default="./train_model" )
    parser.add_argument("--checkpoint", "-c",type = str, default=None )
    args = parser.parse_args()
    return args

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = get_args()
    num_epochs = args.epochs
    batch_size = args.batch_size

    transform = Compose([
        Resize((224, 224)),
        ToTensor(),
    ])

    train_dataset = AnimalDataset(root=r"C:\Users\Admin\Desktop\ki6\xu_ly_anh_so", train=True, transform=transform)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

    model = my_resnet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params= model.parameters())

    for images, labels in train_dataloader:
        print(images.shape)
        print(labels)



if __name__ == '__main__':
    train()