import torch.nn as nn
from torchvision import models

id2label = {
    0: "Glaucoma",
    1: "Catarata",
    2: "Normal",
    3: "Retinopatia diabetica"
}

class EyeDiseaseClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.base = models.resnet18(pretrained=True)

        # Congelar capas base excepto las últimas
        for param in list(self.base.parameters())[:-15]:
            param.requires_grad = False

        # Nueva capa de clasificación
        self.base.fc = nn.Sequential(
            nn.Linear(self.base.fc.in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.base(x)
