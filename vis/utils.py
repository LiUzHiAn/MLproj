import numpy as np
import matplotlib.pyplot as plt
from utils import *


def normalize_image(image):
    """min-max normalization"""
    image_min = np.min(image).copy()
    image_max = np.max(image).copy()
    image = np.clip(image, a_min=image_min, a_max=image_max)
    image = image - image_min
    image = image / (image_max - image_min + 1e-5)

    return image


def plot_images(images, labels, classes, normalize=True):
    n_images = len(images)

    rows = int(np.sqrt(n_images))
    cols = int(np.sqrt(n_images))

    fig = plt.figure(figsize=(15, 15))

    for i in range(rows * cols):

        ax = fig.add_subplot(rows, cols, i + 1)

        image = images[i]

        if normalize:
            image = normalize_image(image)

        ax.imshow(image)
        label = classes[labels[i]]
        ax.set_title(label)
        ax.axis('off')

    plt.show()


def vis_dataset(dataset_train, num_imgs_vis):
    imgs = []
    labels = []

    for i in range(num_imgs_vis):
        img, label = dataset_train[num_imgs_vis + i]
        imgs.append(np.asarray(img))
        labels.append(label)

    plot_images(imgs, labels, classes=INDEX2LABEL)
