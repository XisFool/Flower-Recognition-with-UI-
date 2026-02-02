import torch
import torch.nn as nn
import torchvision.models as models


class FlowerClassifier(nn.Module):
    def __init__(self, num_classes=5, pretrained=True, model_type='resnet50'):
        super(FlowerClassifier, self).__init__()
        self.model_type = model_type

        if model_type == 'resnet50':
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(num_features, num_classes)

        elif model_type == 'resnet18':
            self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(num_features, num_classes)

        elif model_type == 'efficientnet':
            self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier[1] = nn.Linear(num_features, num_classes)

        elif model_type == 'vgg16':
            self.backbone = models.vgg16(weights=models.VGG16_Weights.DEFAULT if pretrained else None)
            num_features = self.backbone.classifier[6].in_features
            self.backbone.classifier[6] = nn.Linear(num_features, num_classes)

        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def forward(self, x):
        return self.backbone(x)

    def unfreeze_layers(self, num_layers_to_unfreeze=0):
        if self.model_type in ['resnet50', 'resnet18']:
            layers = list(self.backbone.children())
            for layer in layers[:-num_layers_to_unfreeze]:
                for param in layer.parameters():
                    param.requires_grad = False
            for layer in layers[-num_layers_to_unfreeze:]:
                for param in layer.parameters():
                    param.requires_grad = True

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

        if self.model_type in ['resnet50', 'resnet18']:
            for param in self.backbone.fc.parameters():
                param.requires_grad = True
        elif self.model_type == 'efficientnet':
            for param in self.backbone.classifier.parameters():
                param.requires_grad = True
        elif self.model_type == 'vgg16':
            for param in self.backbone.classifier.parameters():
                param.requires_grad = True


def create_model(num_classes, model_type='resnet50', pretrained=True):
    model = FlowerClassifier(num_classes=num_classes, pretrained=pretrained, model_type=model_type)
    return model


if __name__ == '__main__':
    model = create_model(num_classes=5, model_type='resnet50', pretrained=True)
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
