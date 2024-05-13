import numpy as np
import torch


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
                    "area": objects["area"][j],  # comes from input data
                    "bbox": output["bboxes"][j],
                }
            )
        annotations = {"annotations": annotations, "image_id": str(item)}
        result = self.image_processor(images=image, annotations=annotations, return_tensors="pt")
        result["pixel_values"] = result["pixel_values"][0]
        result["labels"] = result["labels"][0]
        # result["pixel_values"] = result["pixel_values"].squeeze(0)
        # {'pixel_values': tensor([[[0.3309, 0.3309, 0.3309,  ..., 0.2282, 0.2282, 0.2282],
        #  [0.3309, 0.3309, 0.3309,  ..., 0.2282, 0.2282, 0.2282],
        #  [0.3309, 0.3309, 0.3309,  ..., 0.2282, 0.2282, 0.2282],
        #  ...,
        #  [0.0741, 0.0741, 0.0741,  ..., 0.0741, 0.0741, 0.0741],
        #  [0.0741, 0.0741, 0.0741,  ..., 0.0741, 0.0741, 0.0741],
        #  [0.0741, 0.0741, 0.0741,  ..., 0.0741, 0.0741, 0.0741]],

        # [[0.6078, 0.6078, 0.6078,  ..., 0.3803, 0.3803, 0.3803],
        #  [0.6078, 0.6078, 0.6078,  ..., 0.3803, 0.3803, 0.3803],
        #  [0.6078, 0.6078, 0.6078,  ..., 0.3803, 0.3803, 0.3803],
        #  ...,
        #  [0.2052, 0.2052, 0.2052,  ..., 0.2052, 0.2052, 0.2052],
        #  [0.2052, 0.2052, 0.2052,  ..., 0.2052, 0.2052, 0.2052],
        #  [0.2052, 0.2052, 0.2052,  ..., 0.2052, 0.2052, 0.2052]],

        # [[0.9319, 0.9319, 0.9319,  ..., 0.6356, 0.6356, 0.6356],
        #  [0.9319, 0.9319, 0.9319,  ..., 0.6356, 0.6356, 0.6356],
        #  [0.9319, 0.9319, 0.9319,  ..., 0.6356, 0.6356, 0.6356],
        #  ...,
        #  [0.4265, 0.4265, 0.4265,  ..., 0.4265, 0.4265, 0.4265],
        #  [0.4265, 0.4265, 0.4265,  ..., 0.4265, 0.4265, 0.4265],
        #  [0.4265, 0.4265, 0.4265,  ..., 0.4265, 0.4265, 0.4265]]]), 'labels': [{'image_id': tensor([0]), 'class_labels': tensor([4, 4, 0, 0]), 'boxes': tensor([[0.3590, 0.1432, 0.0774, 0.0552],
        # [0.8892, 0.1209, 0.0604, 0.0297],
        # [0.3012, 0.3596, 0.2630, 0.6535],
        # [0.8921, 0.2848, 0.2125, 0.4254]]), 'area': tensor([  3796.,   1596., 152768.,  81002.]), 'iscrowd': tensor([0, 0, 0, 0]), 'orig_size': tensor([600, 600])}]}
        # image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        # result = {
        #     "pixel_values": torch.tensor(image, dtype=torch.float),
        #     "labels": {
        #         "image_id": torch.tensor(item, dtype=torch.long),
        #         "class_labels": torch.tensor(output["category"], dtype=torch.long),
        #         "boxes": torch.tensor(output["bboxes"], dtype=torch.float),
        #         "area": objects["area"],
        #         "iscrowd": torch.zeros(len(output["bboxes"]), dtype=torch.long),
        #         "orig_size": torch.tensor([image.shape[1], image.shape[2], image.shape[0]], dtype=torch.long),
        #     },
        # }

        return result
