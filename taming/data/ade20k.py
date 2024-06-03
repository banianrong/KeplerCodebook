import os
import numpy as np
import albumentations
from torch.utils.data import Dataset
from PIL import Image
from taming.data.base import ConcatDatasetWithIndex
import cv2

class Ade20kBase(Dataset):
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
        image = (image/127.5 - 1.0).astype(np.float32)

        seg_path = image_path.replace('.jpg', '_seg.png')
        # seg_path = image_path
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

class AdeTrain(Base):
    def __init__(self, size, keys=None):
        super().__init__()
        # root = '/opt/tiger/fnzhan/datasets/ADEChallengeData2016/images/training'
        root = '/data1/lianjunrong/dataset/ADE20K/ADE20K_2021_17_01/images/ADE'
        # with open("data/ade20ktrain.txt", "r") as f:
        with open("/data1/lianjunrong/dataset/ADE20K/ADE20K_2021_17_01/images/ADE/train2.txt", "r") as f:
            relpaths = f.read().splitlines()
        paths = [os.path.join(root, relpath.split(',')[0]) for relpath in relpaths]
        self.data = Ade20kBase(paths=paths, size=size, random_crop=False)
        self.keys = keys

class AdeValidation(Base):
    def __init__(self, size, keys=None):
        super().__init__()
        # root = '/opt/tiger/fnzhan/datasets/ADEChallengeData2016/images/validation'
        root = '/data1/lianjunrong/dataset/ADE20K/ADE20K_2021_17_01/images/ADE'
        # with open("data/ade20kvalidation.txt", "r") as f:
        with open("/data1/lianjunrong/dataset/ADE20K/ADE20K_2021_17_01/images/ADE/test2.txt", "r") as f:
            relpaths = f.read().splitlines()
        paths = [os.path.join(root, relpath.split(',')[0]) for relpath in relpaths]
        self.data = Ade20kBase(paths=paths, size=size, random_crop=False)
        self.keys = keys

class Ade20kTrain(Dataset):
    def __init__(self, size, keys=None, crop_size=None):
        d1 = AdeTrain(size=size, keys=keys)
        # d2 = FFHQTrain(size=size, keys=keys)
        self.data = ConcatDatasetWithIndex([d1])
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


class Ade20kValidation(Dataset):
    # CelebAHQ [0] + FFHQ [1]
    def __init__(self, size, keys=None, crop_size=None):
        d1 = AdeValidation(size=size, keys=keys)
        # d2 = FFHQValidation(size=size, keys=keys)
        self.data = ConcatDatasetWithIndex([d1])
        if crop_size is not None:
            # self.cropper = albumentations.CenterCrop(height=crop_size, width=crop_size)
            # self.resizer = albumentations.Resize(height=crop_size, width=crop_size)
            self.rescaler = albumentations.Resize(height=crop_size, width=crop_size)
            self.transform = albumentations.Compose([self.rescaler, ], additional_targets={"seg": "image"})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        ex, y = self.data[i]

        if hasattr(self, "rescaler"):
            # h, w, _ = ex["image"].shape
            out = self.transform(image=ex["image"], seg=ex['seg'])
            # print('****', out[0].shape)
            ex["image"] = out["image"]
            ex["seg"] = out["seg"]
        ex["class"] = y
        return ex
