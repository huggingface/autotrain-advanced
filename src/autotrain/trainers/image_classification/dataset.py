import numpy as np
import torch


class ImageClassificationDataset:
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
