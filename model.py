from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
import torch.nn as nn
# from torchsummary import summary
import torch

class my_model(nn.Module):
    def __init__(self, num_classes = 3):
        super().__init__()
        self.backbone = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        self.backbone.classifier = nn.Identity()   # bo classifier
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(1280, 256),
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

    model = my_model().to(device)


    # summary(model, input_size=(3,224,224))
