from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import torch.nn as nn
# from torchsummary import summary
import torch

class my_model(nn.Module):
    def __init__(self, num_classes = 36):
        super().__init__()
        self.backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.backbone.classifier = nn.Identity()   # bo classifier
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(1280, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        for p in self.backbone.parameters():
            p.requires_grad = False

        # Unfreeze block cuối
        for p in self.backbone.features[-1].parameters():
            p.requires_grad = True

    def forward(self,x):
        x = self.backbone(x)
        x = self.head(x)
        return x


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = my_model().to(device)


    # summary(model, input_size=(3,224,224))
