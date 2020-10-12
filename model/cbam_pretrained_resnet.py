import torch.nn as nn
    
import torch
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


def initialize_parameters(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data, gain=nn.init.calculate_gain('relu'))
        nn.init.constant_(m.bias.data, 0)

class AttentionResNet(nn.Module):
    model_map = {'resnet18': resnet18,
                 'resnet34': resnet34,
                 'resnet50': resnet50,
                 'resnet101': resnet101,
                 'resnet152': resnet152}

    def __init__(self, model_name, pretrained, num_classes=3):
        super(AttentionResNet, self).__init__()
        self.pretrained = pretrained
        self.num_classes = num_classes
        self.inplanes = 64
        self.dilation = 1

        self.pretrained_model = AttentionResNet.model_map[model_name](pretrained=self.pretrained,
                                                    progress=True)
        self.pretrained_model.fc = nn.Identity()


        self.conv1 = self.pretrained_model.conv1
        self.bn1 = self.pretrained_model.bn1
        self.relu = self.pretrained_model.relu
        self.maxpool = self.pretrained_model.maxpool

        self.ca_first = ChannelAttention(self.inplanes)
        self.sa_first = SpatialAttention()

        self.layer1 = self.pretrained_model.layer1
        self.layer2 = self.pretrained_model.layer2
        self.layer3 = self.pretrained_model.layer3
        self.layer4 = self.pretrained_model.layer4

        self.ca_end = ChannelAttention(2048)
        self.sa_end = SpatialAttention()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)
        self.fc.apply(initialize_parameters)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.ca_first(x) * x
        x = self.sa_first(x) * x
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.ca_end(x) * x
        x = self.sa_end(x) * x
        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x

if __name__ == '__main__':
    md = AttentionResNet("resnet101", pretrained=True, num_classes=3)
    dummy_x = torch.randn(4, 3, 360, 640)
    dummy_out = md(dummy_x)
    print(dummy_out.size(), dummy_out)