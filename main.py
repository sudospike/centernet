from backbone import vgg16
from test import VOCDataset
from  loss import yoloLoss
import torch.nn as nn

def train():

    net = vgg16(pretrained=True)
    net.classifier = nn.Sequential(
        nn.Linear(512*7*7,4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096,1470),
    )

    for m in net.modules():
        if isinstance(m,nn.Linear):
            m.we