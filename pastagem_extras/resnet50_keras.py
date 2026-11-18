# -*- coding: utf-8 -*-
"""
ResNet-50 ImageNet (pesos do Keras Applications) em PyTorch, como extrator.

Usado porque o host dos pesos do torchvision nao estava acessivel. A
arquitetura e a mesma ResNet-50 v1 (estilo Caffe, stride na conv 1x1 do
bloco), com pesos ImageNet. Para quem tiver acesso ao torchvision normal,
embeddings_cnn.py usa ResNet50_Weights do torchvision; os numeros mudam
pouco, mas nao sao identicos.

Pesos: resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5
(github.com/keras-team/keras-applications/releases, tag "resnet").
Pre-processamento do Keras (modo caffe): BGR, subtrai [103.939, 116.779, 123.68].
"""
import h5py
import numpy as np
import torch
import torch.nn as nn

MEDIA_BGR = np.array([103.939, 116.779, 123.68], dtype=np.float32)


class Bloco(nn.Module):
    def __init__(self, cin, meio, cout, stride, com_atalho):
        super().__init__()
        self.c1 = nn.Conv2d(cin, meio, 1, stride=stride)
        self.b1 = nn.BatchNorm2d(meio, eps=1.001e-5)
        self.c2 = nn.Conv2d(meio, meio, 3, padding=1)
        self.b2 = nn.BatchNorm2d(meio, eps=1.001e-5)
        self.c3 = nn.Conv2d(meio, cout, 1)
        self.b3 = nn.BatchNorm2d(cout, eps=1.001e-5)
        self.atalho = None
        if com_atalho:
            self.c0 = nn.Conv2d(cin, cout, 1, stride=stride)
            self.b0 = nn.BatchNorm2d(cout, eps=1.001e-5)
            self.atalho = True

    def forward(self, x):
        s = self.b0(self.c0(x)) if self.atalho else x
        y = torch.relu(self.b1(self.c1(x)))
        y = torch.relu(self.b2(self.c2(y)))
        y = self.b3(self.c3(y))
        return torch.relu(y + s)


class ResNet50Keras(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64, eps=1.001e-5)
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)
        cfg = [(64, 256, 3, 1), (128, 512, 4, 2), (256, 1024, 6, 2), (512, 2048, 3, 2)]
        estagios, cin = [], 64
        for meio, cout, n, stride in cfg:
            blocos = [Bloco(cin, meio, cout, stride, True)]
            blocos += [Bloco(cout, meio, cout, 1, False) for _ in range(n - 1)]
            estagios.append(nn.Sequential(*blocos))
            cin = cout
        self.estagios = nn.ModuleList(estagios)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        for e in self.estagios:
            x = e(x)
        return x.mean(dim=(2, 3))  # media global -> 2048


def _conv(f, nome, conv):
    g = f[nome][nome]
    conv.weight.data = torch.from_numpy(np.array(g["kernel:0"])).permute(3, 2, 0, 1).contiguous()
    conv.bias.data = torch.from_numpy(np.array(g["bias:0"]))


def _bn(f, nome, bn):
    g = f[nome][nome]
    bn.weight.data = torch.from_numpy(np.array(g["gamma:0"]))
    bn.bias.data = torch.from_numpy(np.array(g["beta:0"]))
    bn.running_mean.data = torch.from_numpy(np.array(g["moving_mean:0"]))
    bn.running_var.data = torch.from_numpy(np.array(g["moving_variance:0"]))


def carregar(caminho_h5):
    net = ResNet50Keras()
    with h5py.File(caminho_h5, "r") as f:
        _conv(f, "conv1_conv", net.conv1)
        _bn(f, "conv1_bn", net.bn1)
        for ie, est in enumerate(net.estagios, start=2):
            for ib, bl in enumerate(est, start=1):
                p = "conv%d_block%d_" % (ie, ib)
                if bl.atalho:
                    _conv(f, p + "0_conv", bl.c0)
                    _bn(f, p + "0_bn", bl.b0)
                for j, (c, b) in enumerate(((bl.c1, bl.b1), (bl.c2, bl.b2), (bl.c3, bl.b3)), start=1):
                    _conv(f, p + "%d_conv" % j, c)
                    _bn(f, p + "%d_bn" % j, b)
    net.eval()
    return net


def preparar(img_rgb_uint8):
    """HxWx3 RGB uint8 -> tensor 1x3xHxW no pre-processamento do Keras (caffe)."""
    x = img_rgb_uint8[..., ::-1].astype(np.float32) - MEDIA_BGR  # BGR
    return torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1).unsqueeze(0)
