import torch
import torch.nn as nn
from typing import Optional
import torch.nn.functional as F
from torch.nn.utils import weight_norm


def Fx_Normalization(fx: torch.Tensor, t: float) -> torch.Tensor:
    """
    L2 Normalizaiton for Feature Extraction Output
    :param fx: Output of Feature Extraction
    :param t: temperature coefficients
    :return: normalized feature
    """
    n_feature = fx.shape[1]
    norm = torch.norm(fx, p=2, dim=1).view(-1, 1).repeat(1, n_feature) + 1e-8
    fx = fx / (norm * t)
    return fx


class MLPFeature(nn.Module):
    def __init__(self,
                 input_dim: int,
                 feature_dim: int,
                 t: Optional[float] = 0.1,
                 dataset: Optional[str] = 'MI4C'):
        super(MLPFeature, self).__init__()

        self.block1 = nn.Sequential(nn.Linear(input_dim, input_dim),
                                    nn.BatchNorm1d(input_dim), nn.ELU())
        self.block2 = nn.Sequential(nn.Linear(input_dim, input_dim),
                                    nn.BatchNorm1d(input_dim), nn.ELU())
        self.block3 = nn.Sequential(nn.Linear(input_dim, feature_dim),
                                    nn.BatchNorm1d(feature_dim), nn.ELU())
        self.t = t
        self.dataset = dataset 

    def forward(self, x):
        output = self.block1(x)
        output = self.block2(output)
        output = self.block3(output)
        output = Fx_Normalization(output, self.t)
        return output


class Classifier(nn.Module):
    def __init__(self, input_dim: int, n_classes: int):
        super(Classifier, self).__init__()
        self.fc = weight_norm(nn.Linear(input_dim, n_classes), name="weight")

    def forward(self, x):
        output = self.fc(x)  # logits
        return output


class Head(nn.Module):
    def __init__(self, input_dim: int, dim: int):
        super(Head, self).__init__()
        self.head = nn.Sequential(
            # nn.Linear(input_dim, input_dim),
            # nn.ELU(),
            nn.Linear(input_dim, dim),
        )

    def forward(self, x):
        embedding = self.head(x)
        return F.normalize(embedding)

