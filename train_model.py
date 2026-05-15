from model import my_model
from dataset import AnimalDataset
import torch
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, RandomAffine, ColorJitter


def get_args():
    parser = ArgumentParser("animal classifier")
    parser.add_argument("--batch_size", "-b", type = int, default=128)
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
        RandomAffine(
            degrees=21,
            translate=(0.05, 0.05),
            scale=(0.8, 1.2),
            shear=0.7
        ),
        ColorJitter(
            brightness=0.24,
            contrast=0.5,
            saturation=0.25,
            hue=0.05
        ),
        ToTensor(),
        Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    train_dataset = AnimalDataset(root=r"C:\Users\Admin\Desktop\ki6\xu_ly_anh_so", train=True, transform=transform)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

    model = my_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params= model.parameters())

    num_iters = len(train_dataloader)

    for epoch in range(num_epochs):
        model.train()
        progress_bar = tqdm(train_dataloader, colour='green')
        for iter,  (images, labels) in enumerate(progress_bar):
            images = images.to(device)
            labels = labels.to(device)

            #forward
            outputs = model(images)
            loss_value = criterion(outputs, labels)
            progress_bar.set_description(f'epoch {epoch+1}/{num_epochs}.Iteration {iter+1}/{num_iters}. Loss= {loss_value:.3f} ')


if __name__ == '__main__':
    train()