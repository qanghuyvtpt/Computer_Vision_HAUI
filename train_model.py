from model import my_model
from dataset import AnimalDataset
import torch
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, RandomAffine, ColorJitter
from torch.utils.tensorboard import SummaryWriter
import os
import shutil
from sklearn.metrics import classification_report, accuracy_score



def get_args():
    parser = ArgumentParser("animal classifier")
    parser.add_argument("--batch_size", "-b", type = int, default=256)
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

    train_transform = Compose([
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

    test_transform = Compose([
        Resize((224, 224)),
        ToTensor(),
        Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    if os.path.isdir(args.logging):
        shutil.rmtree(args.logging)
    if not os.path.isdir(args.train_model):
        os.mkdir(args.train_model)
    writer = SummaryWriter(args.logging)    # dua duong dan vao trong

    train_dataset = AnimalDataset(root=r"C:\Users\Admin\Desktop\PythonProject\Xu_ly_anh_so_HAUI\data", train=True, transform=train_transform)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )
    test_dataset = AnimalDataset(root=r"C:\Users\Admin\Desktop\PythonProject\Xu_ly_anh_so_HAUI\data", train=False, transform=test_transform)
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False
    )

    model = my_model().to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4
    )

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"]
        best_acc = checkpoint["best_acc"]
    else:
        start_epoch = 0
        best_acc = -1
    num_iters = len(train_dataloader)

    for epoch in range(start_epoch, num_epochs):
        model.train()
        progress_bar = tqdm(train_dataloader, colour='green')
        for iter,  (images, labels) in enumerate(progress_bar):
            images = images.to(device)
            labels = labels.to(device)

            #forward
            outputs = model(images)
            loss_value = criterion(outputs, labels)
            progress_bar.set_description(f'epoch {epoch+1}/{num_epochs}.Iteration {iter+1}/{num_iters}. Loss= {loss_value:.3f} ')
            writer.add_scalar(tag="train/loss", scalar_value = loss_value, global_step = epoch * num_iters + iter )

            #backward
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()
        # writer.add_scalar("train/loss", loss_value, epoch)

        model.eval()
        all_predictions = []
        all_labels = []
        for iter, (images, label) in enumerate(test_dataloader):
            all_labels.extend(label)
            images = images.to(device)
            label = label.to(device)

            with torch.no_grad():
                preditions = model(images)
                indices = torch.argmax(preditions.cpu(), dim=1)
                all_predictions.extend(indices)
        all_labels = [label.item() for label in all_labels]
        all_predictions = [prediction.item() for prediction in all_predictions]
        acc = accuracy_score(all_labels, all_predictions)
        print(f'epoch: {epoch + 1} ... acc: {acc}')
        writer.add_scalar("Val/Accuracy", acc, epoch)

        if acc > best_acc:
            best_acc = acc
        checkpoint = {
            "model": model.state_dict(),
            "epoch": epoch + 1,
            "optimizer": optimizer.state_dict(),
            "best_acc": best_acc
        }
        torch.save(checkpoint, os.path.join(args.train_model, "last_animal_classifier.pt"))
        if acc == best_acc:
            torch.save(checkpoint, os.path.join(args.train_model, "best_animal_classifier.pt"))




if __name__ == '__main__':
    train()