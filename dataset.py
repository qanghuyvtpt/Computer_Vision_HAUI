import torch
from torch.utils.data import Dataset, DataLoader
import os
import pickle
import cv2
import numpy as np
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torchvision.transforms import ToPILImage

class AnimalDataset(Dataset):
    def __init__(self, root,train = True, transform = None):

        self.root = os.path.join(root, 'Animal_dataset')
        self.transform = transform

        if train:
            self.root = os.path.join(self.root,'train')

        else:
            self.root = os.path.join(self.root,'val')
        categories  = (os.listdir(self.root))
        self.labels = []
        self.images = []

        for i, category in enumerate(categories):
            data_file_path = os.path.join(self.root,category)
            for data_file in os.listdir(data_file_path):
                self.labels.append(i)
                image_file = os.path.join(data_file_path,data_file)
                self.images.append(image_file)
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        image = self.images[idx]
        image = Image.open(image).convert('RGB')
        # image = cv2.imread(image)
        if self.transform:
            image = self.transform(image)
        return image, label

if __name__ == "__main__":
    transform = Compose([
        Resize((224, 224)),
        ToTensor(),
    ])
    dataset = AnimalDataset(root=r"C:\Users\Admin\Desktop\ki6\xu_ly_anh_so", train=True, transform=transform)
    # image, label = dataset.__getitem__(12)
    # img_pil = ToPILImage()(image)
    # # img_pil.show()
    # print(image.shape)
    # print(label)

    train_dataloader = DataLoader(
        dataset=dataset,
        batch_size=4,
        shuffle=True
    )
    for images, labels in train_dataloader:
        print(images.shape)
        print(labels)