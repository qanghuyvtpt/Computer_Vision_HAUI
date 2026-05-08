from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn
from torchsummary import summary
import torch

class my_resnet(nn.Module):
    def __init__(self, num_classes = 2):
        super().__init__()
        self.backbone = resnet50(weights = ResNet50_Weights.DEFAULT)
        self.backbone.fc = nn.Identity()    # bo classifier
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        for p in self.backbone.parameters():
            p.requires_grad = False
    def forward(self,x):
        x = self.backbone(x)
        x = self.head(x)
        return x


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = my_resnet().to(device)


    summary(model, input_size=(3,224,224))
