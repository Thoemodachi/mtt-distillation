import torch.nn as nn
import torch.nn.functional as F
import torch
from facenet_pytorch import InceptionResnetV1  # pretrained FaceNet
from torchvision import models # pretrained vggface
from arcface import ArcFaceLoss  # pretrained ArcFace

# =============================================================================
# Acknowledgements
# -----------------------------------------------------------------------------
# FaceNet model and MTCNN provided by facenet‑pytorch (Tim Esler),
# with pretrained weights ported from David Sandberg’s TensorFlow implementation :contentReference[oaicite:0]{index=0}.
#
# Original FaceNet paper:
# Schroff, Kalenichenko & Philbin (2015), “FaceNet: A Unified Embedding for Face Recognition and Clustering” (CVPR) :contentReference[oaicite:1]{index=1}.
#
# VGGFace2 dataset reference:
# Cao, Shen, Xie, Parkhi & Zisserman (2017), “VGGFace2: A dataset for recognising faces across pose and age” (FG) :contentReference[oaicite:2]{index=2}.
#
# ArcFace implementation and library references:
# Deng, Guo, Xue & Zafeiriou (2019), “ArcFace: Additive Angular Margin Loss for Deep Face Recognition” (CVPR) :contentReference[oaicite:3]{index=3}.
#
# ArcFace PyTorch implementations (e.g., GOKORURI007’s pytorch_arcface, shyhyawJou’s, and ronghuaiyang’s repos).
# InsightFace project provides the official origin and codebase :contentReference[oaicite:4]{index=4}.
# =============================================================================

# FaceNet: outputs 512-D embeddings
class FaceNet(nn.Module):
    def __init__(self, num_classes=None, pretrained='vggface2', classify=False):
        super().__init__()
        self.backbone = InceptionResnetV1(pretrained=pretrained, classify=classify, num_classes=num_classes)
    def forward(self, x):
        return self.backbone(x)

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

# ArcFace: ResNet-50 backbone + ArcFace margin loss layer
class ArcFaceNet(nn.Module):
    def __init__(self, num_classes, embedding_size=512, margin=0.5, scale=64):
        super().__init__()
        self.backbone = models.resnet50(pretrained=True)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, embedding_size)
        self.arcface = ArcFaceLoss(embedding_size, num_classes, margin=margin, scale=scale)

    def forward(self, x, labels=None):
        emb = self.backbone(x)
        emb = F.normalize(emb)
        if labels is not None:
            return self.arcface(emb, labels)   # returns logits via angular margin
        return emb
