import torch
from torchvision.transforms import transforms
import PIL

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EXP_NAME = 'pretrained_vgg19'

PRETRAIN = True
NUM_EPOCHS = 80
START_LR = 1e-3
N_IMAGES_VIS = 25
BATCH_SIZE = 32
# resize_size = 256
spatial_size = 224
pretrained_means = [0.485, 0.456, 0.406]
pretrained_stds = [0.229, 0.224, 0.225]
NUM_WORKERS = 32

train_transforms = transforms.Compose([
    transforms.Resize((spatial_size, spatial_size)),
    transforms.ColorJitter(hue=.05, saturation=.05),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20, resample=PIL.Image.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(mean=pretrained_means,
                         std=pretrained_stds)
])

test_transforms = transforms.Compose([
    transforms.Resize((spatial_size, spatial_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=pretrained_means,
                         std=pretrained_stds)
])
