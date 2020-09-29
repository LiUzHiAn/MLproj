import torch
import torchvision.models as models
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
import torchvision.transforms as T
from PIL import Image
import torch.nn as nn
from utils import initialize_parameters


class MyResNet(nn.Module):
    model_map = {'resnet18': resnet18,
                 'resnet34': resnet34,
                 'resnet50': resnet50,
                 'resnet101': resnet101,
                 'resnet152': resnet152}

    def __init__(self, model_name, pretrained, num_classes):
        super(MyResNet, self).__init__()
        assert MyResNet.model_map.get(model_name) is not None, \
            model_name + "is not avaliable!"
        self.pretrained = pretrained
        self.num_classes = num_classes
        self.model = MyResNet.model_map[model_name](pretrained=self.pretrained,
                                                    progress=True)
        # self._set_parameter_requires_grad()

        # resnet最后一个conv的输出是后，会flatten成一个 [bs,2048]的feature vectors
        # 换掉最后一个fc层
        num_ftrs = self.model.fc.in_features
        # self.model.fc = nn.Sequential(
        #     nn.Linear(2048, 512),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(p=0.3),
        #     nn.Linear(512, self.num_classes)
        # )
        self.model.fc = nn.Linear(num_ftrs, self.num_classes)

        self.model.fc.apply(initialize_parameters)

    def _set_parameter_requires_grad(self):
        if self.pretrained:
            for param in self.model.parameters():
                param.requires_grad = False

    def transfer_learning(self, filename):
        trans = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        input_image = Image.open(filename)
        # dummy = T.ToTensor()(input_image)
        input_tensor = trans(input_image)
        input_batch = input_tensor.unsqueeze(0)

        with torch.no_grad():
            test_model = self.model.eval()
            output = test_model(input_batch)
        # Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
        # print(output[0])
        # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
        print(torch.argmax(torch.nn.functional.softmax(output[0], dim=0)))

    def forward(self, x):
        res = self.model(x)
        return res


if __name__ == '__main__':
    md = MyResNet("resnet50", pretrained=True, num_classes=3)
    # md.transfer_learning(filename="../inference/images/21_2.jpg")
    dummy_x = torch.randn(4, 3, 360, 640)
    dummy_out = md(dummy_x)
    print(dummy_out.size(), dummy_out)
