import numpy as np
import torch


class ImageClassificationDataset:
    """
    A custom dataset class for image classification tasks.

    Args:
        data (list): A list of data samples, where each sample is a dictionary containing image and target information.
        transforms (callable): A function/transform that takes in an image and returns a transformed version.
        config (object): A configuration object containing the column names for images and targets.

    Attributes:
        data (list): The dataset containing image and target information.
        transforms (callable): The transformation function to be applied to the images.
        config (object): The configuration object with image and target column names.

    Methods:
        __len__(): Returns the number of samples in the dataset.
        __getitem__(item): Retrieves the image and target at the specified index, applies transformations, and returns them as tensors.

    Example:
        dataset = ImageClassificationDataset(data, transforms, config)
        image, target = dataset[0]
    """

    def __init__(self, data, transforms, config):
        self.data = data
        self.transforms = transforms
        self.config = config

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        image = self.data[item][self.config.image_column]
        target = int(self.data[item][self.config.target_column])

        image = self.transforms(image=np.array(image.convert("RGB")))["image"]
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)

        return {
            "pixel_values": torch.tensor(image, dtype=torch.float),
            "labels": torch.tensor(target, dtype=torch.long),
        }
