import numpy as np


class ObjectDetectionDataset:
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
