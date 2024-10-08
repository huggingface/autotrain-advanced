import numpy as np


class ObjectDetectionDataset:
    """
    A dataset class for object detection tasks.

    Args:
        data (list): A list of data entries where each entry is a dictionary containing image and object information.
        transforms (callable): A function or transform to apply to the images and bounding boxes.
        image_processor (callable): A function or processor to convert images and annotations into the desired format.
        config (object): A configuration object containing column names for images and objects.

    Attributes:
        data (list): The dataset containing image and object information.
        transforms (callable): The transform function to apply to the images and bounding boxes.
        image_processor (callable): The processor to convert images and annotations into the desired format.
        config (object): The configuration object with column names for images and objects.

    Methods:
        __len__(): Returns the number of items in the dataset.
        __getitem__(item): Retrieves and processes the image and annotations for the given index.

    Example:
        dataset = ObjectDetectionDataset(data, transforms, image_processor, config)
        image_data = dataset[0]
    """

    def __init__(self, data, transforms, image_processor, config):
        self.data = data
        self.transforms = transforms
        self.image_processor = image_processor
        self.config = config

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        image = self.data[item][self.config.image_column]
        objects = self.data[item][self.config.objects_column]
        output = self.transforms(
            image=np.array(image.convert("RGB")), bboxes=objects["bbox"], category=objects["category"]
        )
        image = output["image"]
        annotations = []
        for j in range(len(output["bboxes"])):
            annotations.append(
                {
                    "image_id": str(item),
                    "category_id": output["category"][j],
                    "iscrowd": 0,
                    "area": objects["bbox"][j][2] * objects["bbox"][j][3],  # [x, y, w, h
                    "bbox": output["bboxes"][j],
                }
            )
        annotations = {"annotations": annotations, "image_id": str(item)}
        result = self.image_processor(images=image, annotations=annotations, return_tensors="pt")
        result["pixel_values"] = result["pixel_values"][0]
        result["labels"] = result["labels"][0]
        return result
