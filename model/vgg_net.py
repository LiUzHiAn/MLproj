import torch
import torchvision.models as models
from torchvision.models import vgg16, vgg19
import torchvision.transforms as T
from PIL import Image
import torch.nn as nn


def initialize_parameters(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data, gain=nn.init.calculate_gain('relu'))
        nn.init.constant_(m.bias.data, 0)


class MyVGGNet(nn.Module):
    model_map = {'vgg16': vgg16,
                 'vgg19': vgg19,
                 }

    def __init__(self, model_name, pretrained, num_classes):
        super(MyVGGNet, self).__init__()
        assert MyVGGNet.model_map.get(model_name) is not None, \
            model_name + "is not avaliable!"
        self.pretrained = pretrained
        self.num_classes = num_classes
        self.model = MyVGGNet.model_map[model_name](pretrained=self.pretrained,
                                                    progress=True)

        num_ftrs = self.model.classifier[0].in_features
        self.model.classifier = nn.Linear(num_ftrs, self.num_classes)

        self.model.classifier.apply(initialize_parameters)

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
    md = MyVGGNet("vgg19", pretrained=True, num_classes=3)
    # md.transfer_learning(filename="../inference/images/21_2.jpg")
    dummy_x = torch.randn(4, 3, 360, 640)
    dummy_out = md(dummy_x)
    print(dummy_out.size(), dummy_out)
