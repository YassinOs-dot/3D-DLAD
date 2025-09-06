import os
from PIL import Image
from torchvision import transforms
import glob
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
from utils.py import *
DATASETS_PATH = "/content/drive/MyDrive/"
CLASS_NAME = "cookie"


class MVTec3D(Dataset):
    def __init__(self, split, img_size):
        self.IMAGENET_MEAN = [0.485, 0.456, 0.406]
        self.IMAGENET_STD = [0.229, 0.224, 0.225]
        self.cls = CLASS_NAME
        self.size = img_size
        self.img_path = os.path.join(DATASETS_PATH, self.cls, split)
        self.rgb_transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)
        ])


class DATATRAIN(MVTec3D):
    def __init__(self, img_size):
        super().__init__(split="train", img_size=img_size)
        self.img_paths, self.labels = self.load_dataset()

    def load_dataset(self):
        img_tot_paths = []
        tot_labels = []
        rgb_paths = glob.glob(os.path.join(self.img_path, 'good', 'rgb', "*.png"))
        tiff_paths = glob.glob(os.path.join(self.img_path, 'good', 'xyz', "*.tiff"))
        rgb_paths.sort()
        tiff_paths.sort()
        sample_paths = list(zip(rgb_paths, tiff_paths))
        img_tot_paths.extend(sample_paths)
        tot_labels.extend([0] * len(sample_paths))
        return img_tot_paths, tot_labels

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, label = self.img_paths[idx], self.labels[idx]
        rgb_path, tiff_path = img_path
        img = Image.open(rgb_path).convert('RGB')
        img = self.rgb_transform(img)

        organized_pc = read_tiff_organized_pc(tiff_path)
        depth_map_3channel = np.repeat(
            organized_pc_to_depth_map(organized_pc)[:, :, np.newaxis], 3, axis=2
        )
        resized_depth_map_3channel = resize_organized_pc(depth_map_3channel)
        resized_organized_pc = resize_organized_pc(organized_pc)
        return (img, resized_organized_pc, resized_depth_map_3channel), label


class DATATEST(MVTec3D):
    def __init__(self, img_size):
        super().__init__(split="test", img_size=img_size)
        self.gt_transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])
        self.img_paths, self.gt_paths, self.labels = self.load_dataset()

    def load_dataset(self):
        img_tot_paths, gt_tot_paths, tot_labels = [], [], []
        defect_types = os.listdir(self.img_path)

        for defect_type in defect_types:
            if defect_type == 'good':
                rgb_paths = glob.glob(os.path.join(self.img_path, defect_type, 'rgb', "*.png"))
                tiff_paths = glob.glob(os.path.join(self.img_path, defect_type, 'xyz', "*.tiff"))
                rgb_paths.sort()
                tiff_paths.sort()
                sample_paths = list(zip(rgb_paths, tiff_paths))
                img_tot_paths.extend(sample_paths)
                gt_tot_paths.extend([0] * len(sample_paths))
                tot_labels.extend([0] * len(sample_paths))
            else:
                rgb_paths = glob.glob(os.path.join(self.img_path, defect_type, 'rgb', "*.png"))
                tiff_paths = glob.glob(os.path.join(self.img_path, defect_type, 'xyz', "*.tiff"))
                gt_paths = glob.glob(os.path.join(self.img_path, defect_type, 'gt', "*.png"))
                rgb_paths.sort()
                tiff_paths.sort()
                gt_paths.sort()
                sample_paths = list(zip(rgb_paths, tiff_paths))
                img_tot_paths.extend(sample_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(sample_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Mismatch test vs gt pair!"
        return img_tot_paths, gt_tot_paths, tot_labels

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label = self.img_paths[idx], self.gt_paths[idx], self.labels[idx]
        rgb_path, tiff_path = img_path
        img_original = Image.open(rgb_path).convert('RGB')
        img = self.rgb_transform(img_original)

        organized_pc = read_tiff_organized_pc(tiff_path)
        depth_map_3channel = np.repeat(
            organized_pc_to_depth_map(organized_pc)[:, :, np.newaxis], 3, axis=2
        )
        resized_depth_map_3channel = resize_organized_pc(depth_map_3channel)
        resized_organized_pc = resize_organized_pc(organized_pc)

        if gt == 0:
            gt = torch.zeros([1, resized_depth_map_3channel.shape[0], resized_depth_map_3channel.shape[1]])
        else:
            gt = Image.open(gt).convert('L')
            gt = self.gt_transform(gt)
            gt = torch.where(gt > 0.5, 1., .0)

        return (img, resized_organized_pc, resized_depth_map_3channel), gt[:1], label


def DATALOADER(split, img_size):
    """DataLoader fixed for 'cookie' only"""
    if split == 'train':
        dataset = DATATRAIN(img_size=img_size)
    elif split == 'test':
        dataset = DATATEST(img_size=img_size)
    else:
        raise ValueError("split must be 'train' or 'test'")

    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False, pin_memory=True)
