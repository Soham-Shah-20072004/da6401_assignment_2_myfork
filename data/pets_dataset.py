"""Dataset skeleton for Oxford-IIIT Pet.
"""

from torch.utils.data import Dataset
import os
import xml.etree.ElementTree as ET
from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class OxfordIIITPetDataset(Dataset):
    """Oxford-IIIT Pet multi-task dataset loader skeleton."""

    def __init__(self, root_dir: str, split: str = "train", transform=None):
        """
        Initialize the OxfordIIITPetDataset.
        Args:
            root_dir: Path to the dataset root directory.
            split: Split to use ("train" or "test").
            transform: Optional Albumentations transform.
        """
        super().__init__()
        self.root_dir = root_dir
        self.images_dir = os.path.join(root_dir, "images")
        self.masks_dir = os.path.join(root_dir, "annotations", "trimaps")
        self.xml_dir  = os.path.join(root_dir, "annotations", "xmls")
    
        self.transform = transform

        # ---------------------------------------------
        # split id
        # if split == "train":
        #     split_id = 1
        # else:
        #     split_id = 2

        # list_path = os.path.join(root_dir, "annotations", "list.txt")
        # self.samples = []  # each entry: (image_name, class_idx)
        # with open(list_path, "r") as f:
        #     for line in f:
        #         line = line.strip()
        #         if not line or line.startswith("#"):
        #             continue
        #         parts = line.split()
        #         name      = parts[0]          # e.g. "Abyssinian_1"
        #         class_id  = int(parts[1]) - 1 # 1-indexed → 0-indexed
        #         img_split = int(parts[3])     # 1=train, 2=test

        #         if img_split != split_id:
        #             continue # skip those images which dont match with the current mode, for split_id = 1, we skip those images whose img_split is 2
        
        # ----------------------------------------

        if split == "train":
            list_path = os.path.join(root_dir, "annotations", "trainval.txt")
        else:
            list_path = os.path.join(root_dir, "annotations", "test.txt")
        self.samples = []

        with open(list_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                name      = parts[0]
                class_id  = int(parts[1]) - 1        
                xml_path = os.path.join(self.xml_dir, name + ".xml") # path to xml_dir + imagename.xml is the path to the xml file
                has_bbox = os.path.exists(xml_path)
                if split == "train" and not has_bbox:
                    continue # skip train images without bounding box

                self.samples.append((name, class_id))
                # now we have the image name, class id, and xml path
                # we need to extract the bounding box from the xml file
                # and then add the image name, class id, and xml path to the self.samples list

        if self.transform is None:
            self.transform = A.Compose([
                A.Resize(224, 224),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ], bbox_params=A.BboxParams(
                format="pascal_voc", # because the bounding boxes are in xmin, ymin, xmax, ymax format
                label_fields=["bbox_labels"]
            ))

    def __len__(self):
        return len(self.samples)  

    def __getitem__(self, idx):
        name, class_id = self.samples[idx]

        # get the image path
        image_path = os.path.join(self.images_dir, name + ".jpg")
        # image object
        image = Image.open(image_path).convert("RGB")
        image = np.array(image)
        orig_h, orig_w = image.shape[:2] # get the height and width of the original image

        # get the mask path
        mask_path = os.path.join(self.masks_dir, name + ".png")
        # mask object
        mask = Image.open(mask_path)
        mask = np.array(mask)
        mask = mask -1 # cross entropy loss needs 0,1,2 format

        # get the bounding box
        xml_path = os.path.join(self.xml_dir, name + ".xml")
        if os.path.exists(xml_path):
            xmin, ymin, xmax, ymax = self._parse_xml(xml_path)
        else:
            xmin, ymin, xmax, ymax = 0, 0, orig_w, orig_h
        
        transformed = self.transform(
            image=image,
            mask=mask,
            bboxes=[[xmin, ymin, xmax, ymax]],
            bbox_labels=[0]                          # dummy label, we use class_id
        )

        # bbox_labels=[0] is a dummy label required by albumentations API — we don't actually use it since we have class_id.

        image = transformed["image"]                 # [3, 224, 224] tensor
        mask  = transformed["mask"].long()           # [224, 224] tensor



        # get transformed bbox → convert corners to cx,cy,w,h
        bbox         = transformed["bboxes"][0]      # [xmin,ymin,xmax,ymax] scaled
        xmin, ymin, xmax, ymax = bbox
        x_center = (xmin + xmax) / 2
        y_center = (ymin + ymax) / 2
        width    = xmax - xmin
        height   = ymax - ymin
        bbox_tensor = torch.tensor(
            [x_center, y_center, width, height],
            dtype=torch.float32
        )

        return {
            "image":      image,                     # [3, 224, 224]
            "label":      torch.tensor(class_id),    # scalar
            "bbox":       bbox_tensor,               # [4]
            "mask":       mask                       # [224, 224]
        }


    def _parse_xml(self, xml_path: str):
        """Parse bounding box from VOC-style XML."""
        tree = ET.parse(xml_path)
        root = tree.getroot()
        bndbox = root.find(".//bndbox")
        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)
        return xmin, ymin, xmax, ymax

    pass

