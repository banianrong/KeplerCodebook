import os
import numpy as np
import albumentations
from torch.utils.data import Dataset
from PIL import Image
import cv2
from taming.data.base import ConcatDatasetWithIndex
# ImagePaths, NumpyPaths,


class FacesBase(Dataset):
    def __init__(self, paths, size=None, random_crop=False, labels=None):
        self.size = size
        self.random_crop = random_crop

        self.labels = dict() if labels is None else labels
        self.labels["file_path_"] = paths
        self._length = len(paths)

        if self.size is not None and self.size > 0:
            # self.rescaler = albumentations.SmallestMaxSize(max_size = self.size)

            self.im_rescaler = albumentations.SmallestMaxSize(max_size=self.size,
                                                              interpolation=cv2.INTER_CUBIC)
            self.seg_rescaler = albumentations.SmallestMaxSize(max_size=self.size,
                                                               interpolation=cv2.INTER_NEAREST)

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = dict()
        example["image"], example['seg'] = self.preprocess_image(self.labels["file_path_"][i])
        for k in self.labels:
            example[k] = self.labels[k][i]
        return example

    def preprocess_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image = np.array(image)
        # image = np.transpose(image, (1,2,0))
        # image = Image.fromarray(image, mode="RGB")
        # image = np.array(image).astype(np.uint8)
        image = self.im_rescaler(image=image)["image"]
        image = (image / 127.5 - 1.0).astype(np.float32)

        seg_path = os.path.dirname(image_path).replace('CelebA-HQ-img', 'CelebAMask-HQ-mask-anno/all_parts_except_glasses') \
                   + '/' + os.path.basename(image_path).split('.')[0].strip().zfill(5) + '.png'

        seg = Image.open(seg_path)
        seg = np.array(seg)
        seg = self.seg_rescaler(image=seg)["image"]

        return image, seg



class Base(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.data = None
        self.keys = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        example = self.data[i]
        ex = {}
        if self.keys is not None:
            for k in self.keys:
                ex[k] = example[k]
        else:
            ex = example
        return ex

class CelebAHQTrain(Base):
    def __init__(self, size, keys=None):
        super().__init__()
        root = '/opt/tiger/fnzhan/datasets/CelebAMask-HQ/CelebA-HQ-img'
        with open("data/celebahqtrain.txt", "r") as f:
            relpaths = f.read().splitlines()
        paths = [os.path.join(root, relpath.split(',')[0]) for relpath in relpaths]
        self.data = FacesBase(paths=paths, size=size, random_crop=False)
        self.keys = keys

class CelebAHQValidation(Base):
    def __init__(self, size, keys=None):
        super().__init__()
        root = '/opt/tiger/fnzhan/datasets/CelebAMask-HQ/CelebA-HQ-img'
        with open("data/celebahqvalidation.txt", "r") as f:
            relpaths = f.read().splitlines()
        paths = [os.path.join(root, relpath.split(',')[0]) for relpath in relpaths]
        self.data = FacesBase(paths=paths, size=size, random_crop=False)
        self.keys = keys

class FacesHQTrain(Dataset):
    def __init__(self, size, keys=None, crop_size=None, coord=False):
        d1 = CelebAHQTrain(size=size, keys=keys)
        # d2 = FFHQTrain(size=size, keys=keys)
        self.data = ConcatDatasetWithIndex([d1])
        self.coord = coord
        if crop_size is not None:
            self.cropper = albumentations.RandomCrop(height=crop_size, width=crop_size)
            self.flip = albumentations.HorizontalFlip(always_apply=False, p=0.5)
            self.transform = albumentations.Compose([self.cropper, self.flip, ],
                                                    additional_targets={"seg": "image"})
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        ex, y = self.data[i]
        if hasattr(self, "cropper"):
            h, w, _ = ex["image"].shape
            out = self.transform(image=ex["image"], seg=ex['seg'])
            ex["image"] = out["image"]
            ex["seg"] = out["seg"]
        ex["class"] = y
        return ex


class FacesHQValidation(Dataset):
    # CelebAHQ [0] + FFHQ [1]
    def __init__(self, size, keys=None, crop_size=None, coord=False):
        d1 = CelebAHQValidation(size=size, keys=keys)
        # d2 = FFHQValidation(size=size, keys=keys)
        self.data = ConcatDatasetWithIndex([d1])
        self.coord = coord
        if crop_size is not None:
            # self.cropper = albumentations.CenterCrop(height=crop_size, width=crop_size)
            self.rescaler = albumentations.Resize(height=crop_size, width=crop_size)
            self.transform = albumentations.Compose([self.rescaler, ],
                                                    additional_targets={"seg": "image"})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        ex, y = self.data[i]

        if hasattr(self, "cropper"):
            h, w, _ = ex["image"].shape
            out = self.transform(image=ex["image"], seg=ex['seg'])
            ex["image"] = out["image"]
            ex["seg"] = out["seg"]
        ex["class"] = y
        return ex
