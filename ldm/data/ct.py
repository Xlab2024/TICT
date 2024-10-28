import os, yaml, pickle, shutil, tarfile, glob
import cv2
import albumentations
import PIL
import numpy as np
import sys
import torchvision.transforms.functional as TF
import torch
import cv2
import astra
import scipy.io
from omegaconf import OmegaConf
from functools import partial
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, Subset

import taming.data.utils as tdu
from taming.data.imagenet import str_to_indices, give_synsets_from_indices, download, retrieve
from taming.data.imagenet import ImagePaths

from ldm.modules.image_degradation import degradation_fn_bsr, degradation_fn_bsr_light
from torchvision import transforms
sys.path.append('...')
from physics.ct import CT

def norm01(data):
    max=np.max(data)
    min=np.min(data)
    ran=max-min
    data01=(data-min)/ran*1.0
    return data01

class CTPistonSRckpt_BP(Dataset):
    def __init__(self, 
                 txt_file,
                 data_root,
                 size=None,
                 degradation=None, 
                 downscale_f=4):

        self.data_paths = txt_file
        self.data_root = data_root
        self.size = size
        self.degradation = degradation
        self.downscale_f=downscale_f
        self.Nview = 2
        with open(self.data_paths, "r") as f:
            self.image_paths = f.read().splitlines()
        self._length = len(self.image_paths)
        self.labels = {
            "relative_file_path_": [l for l in self.image_paths],
            "file_path_": [os.path.join(self.data_root, l)
                           for l in self.image_paths],
        }
        if degradation == "radon":
            self.angles = np.linspace(0, np.pi, 180, endpoint=False)
            self.radon = CT(img_width=size, radon_view=self.Nview, circle=False)

    def __len__(self):
        return self._length
    
    def __getitem__(self, i):
        example = dict((k, self.labels[k][i]) for k in self.labels)
        if example["file_path_"][-3:]=='png':
            image = cv2.imread(example["file_path_"])
            image = image.astype(np.float32)
            image = norm01(image)
            image1 = image[:,:,1]
        else:
            image = np.load(example["file_path_"])  
    
        if self.degradation == "radon":
            image3d=image1[np.newaxis,:]
            image4d=image3d[np.newaxis,:]
            image4d=torch.from_numpy(image4d)
            bpimage4d = self.radon.Getbp(image4d).numpy()
            bpimage4d = norm01(bpimage4d)/0.5 - 1.0
            cimage4d = bpimage4d
            fbpimage3d=cimage4d.squeeze().squeeze()
            example["FBP_image"] = fbpimage3d
        example["image"] = image/0.5 - 1.0
        return example


class CTPistonSRTrainckpt_BP(CTPistonSRckpt_BP):
    def __init__(self, **kwargs):
        super().__init__(txt_file="data/CT/train_png.txt", data_root="data/CT/all_piston256_png",**kwargs)


class CTPistonSRValidationckpt_BP(CTPistonSRckpt_BP):
    def __init__(self, **kwargs):
        super().__init__(txt_file="data/CT/val_png.txt", data_root="data/CT/all_piston256_png", **kwargs)