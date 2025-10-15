import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision import models  # pretrained models

try:
    from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
except ImportError:  # torchvision < 0.13 fallback
    from torchvision.models import mobilenet_v2

    MobileNet_V2_Weights = None  # type: ignore

try:
    from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
except ImportError:  # efficientnet not available in older torchvision
    efficientnet_b0 = None  # type: ignore
    EfficientNet_B0_Weights = None  # type: ignore

MODEL_ACKNOWLEDGEMENTS = {
    "VGGFace": (
        "VGG-Face architecture from Parkhi et al., Deep Face Recognition (BMVC 2015); "
        "pretrained weights via torchvision."
    ),
    "MobileNetV2": (
        "MobileNetV2 architecture from Sandler et al., MobileNetV2: Inverted Residuals "
        "and Linear Bottlenecks (CVPR 2018); pretrained weights via torchvision."
    ),
    "EfficientNetB0": (
        "EfficientNet-B0 architecture from Tan and Le, EfficientNet: Rethinking Model "
        "Scaling for Convolutional Neural Networks (ICML 2019); pretrained weights via "
        "torchvision."
    ),
}

# VGG‑Face: use torchvision VGG16 and optionally add custom classifier
class VGGFace(nn.Module):
    def __init__(self, embedding_size=512, num_classes=None):
        super().__init__()
        vgg = models.vgg16_bn(pretrained=True)
        self.features = vgg.features
        self.avgpool = vgg.avgpool
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, embedding_size),
            nn.BatchNorm1d(embedding_size),
        )
        self.class_logits = nn.Linear(embedding_size, num_classes) if num_classes is not None else None

    def forward(self, x):
        h = self.features(x)
        h = self.avgpool(h)
        h = torch.flatten(h, 1)
        emb = self.classifier(h)
        if self.class_logits:
            return self.class_logits(emb)
        return F.normalize(emb)


class MobileNetV2Face(nn.Module):
    def __init__(self, embedding_size=256, num_classes=None, pretrained=True):
        super().__init__()
        if MobileNet_V2_Weights is None:
            backbone = mobilenet_v2(pretrained=pretrained)
        else:
            weights = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = mobilenet_v2(weights=weights)
        dropout_p = getattr(backbone.classifier[0], "p", 0.2)
        self.features = backbone.features
        self.pool = backbone.avgpool
        self.embedding_head = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(backbone.last_channel, embedding_size),
            nn.BatchNorm1d(embedding_size),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(embedding_size, num_classes) if num_classes is not None else None

    def forward(self, x):
        h = self.features(x)
        h = self.pool(h)
        h = torch.flatten(h, 1)
        emb = self.embedding_head(h)
        if self.classifier:
            return self.classifier(emb)
        return F.normalize(emb)


class EfficientNetB0Face(nn.Module):
    def __init__(self, embedding_size=256, num_classes=None, pretrained=True):
        super().__init__()
        if efficientnet_b0 is None:
            raise RuntimeError("EfficientNet models are unavailable in this torchvision version.")
        if EfficientNet_B0_Weights is None:
            backbone = efficientnet_b0(pretrained=pretrained)
        else:
            weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = efficientnet_b0(weights=weights)
        dropout_p = backbone.classifier[0].p if hasattr(backbone.classifier[0], "p") else 0.2
        in_features = backbone.classifier[1].in_features
        self.features = backbone.features
        self.pool = backbone.avgpool
        self.embedding_head = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features, embedding_size),
            nn.BatchNorm1d(embedding_size),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(embedding_size, num_classes) if num_classes is not None else None

    def forward(self, x):
        h = self.features(x)
        h = self.pool(h)
        h = torch.flatten(h, 1)
        emb = self.embedding_head(h)
        if self.classifier:
            return self.classifier(emb)
        return F.normalize(emb)
