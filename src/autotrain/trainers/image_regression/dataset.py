import numpy as np
import torch


class ImageRegressionDataset:
    """
    A dataset class for image regression tasks.

    Args:
        data (list): A list of data points where each data point is a dictionary containing image and target information.
        transforms (callable): A function/transform that takes in an image and returns a transformed version.
        config (object): A configuration object that contains the column names for images and targets.

    Attributes:
        data (list): The input data.
        transforms (callable): The transformation function.
        config (object): The configuration object.

    Methods:
        __len__(): Returns the number of data points in the dataset.
        __getitem__(item): Returns a dictionary containing the transformed image and the target value for the given index.
    """

    def __init__(self, data, transforms, config):
        self.data = data
        self.transforms = transforms
        self.config = config

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        image = self.data[item][self.config.image_column]
        target = self.data[item][self.config.target_column]

        image = self.transforms(image=np.array(image.convert("RGB")))["image"]
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)

        return {
            "pixel_values": torch.tensor(image, dtype=torch.float),
            "labels": torch.tensor(target, dtype=torch.float),
        }
