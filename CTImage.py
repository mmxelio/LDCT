import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
import os
import re
from collections import defaultdict
from PIL import Image
import glob
import numpy as np

class CTImage(Dataset):
    def __init__(self,patch_n=None, patch_size = None,root_dir="../data/TrainSet/",transforms=transforms.ToTensor()):
        self.root_dir = root_dir
        self.transforms = transforms
        self.patch_n = patch_n
        self.patch_size = patch_size
        self.quarter_dose = os.path.join(self.root_dir,"Quarter_Dose")
        self.full_dose = os.path.join(self.root_dir,"Full_Dose")
        self.quarter_files = sorted(glob.glob(os.path.join(self.quarter_dose,"*","*.png")))
        self.full_files = sorted(glob.glob(os.path.join(self.full_dose,"*","*.png")))

    def group_files(files):
        file_dict = defaultdict(list)
        pattern = re.compile(r'L(\d+)_QD_(\d+)_.*')
        for file in files:
            match = pattern.search(file)
            if match:
                pid = match.group(1)
                file_dict[pid].append(file)
        return file_dict

    def __len__(self):
        return len(self.quarter_files)


    def __getitem__(self, index):
        qd_name = os.path.basename(self.quarter_files[index])
        fd_name = os.path.basename(self.full_files[index])

        # pattern1 = re.compile(r'L(\d+)_QD_(\d+)_.*')
        # pattern2 = re.compile(r'L(\d+)_FD_(\d+)_.*')
        # match1 = pattern1.search(qd_name)
        # match2 = pattern2.search(fd_name)
        # assert(match1.group(1) == match2.group(1))
        # assert(qd_name.split('.')[3] == fd_name.split('.')[3]) 
        qd_img = Image.open(self.quarter_files[index]).convert('L')
        fd_img = Image.open(self.full_files[index]).convert('L')
        qd_img = self.transforms(qd_img)
        fd_img = self.transforms(fd_img)
        if self.patch_size:
            qd_patch,fd_patch = get_patch(qd_img,fd_img,self.patch_n,self.patch_size)
            return (qd_patch,fd_patch,qd_name,fd_name)
        return (qd_img,fd_img,qd_name,fd_name)


def get_patch(full_input_img, full_target_img, patch_n, patch_size=64):
    assert full_input_img.shape == full_target_img.shape
    patch_input_imgs = []
    patch_target_imgs = []
    _, h, w = full_input_img.shape
    new_h, new_w = patch_size, patch_size
    for _ in range(patch_n):
        top = np.random.randint(0, h-new_h)
        left = np.random.randint(0, w-new_w)
        patch_input_img = full_input_img[:,top:top+new_h, left:left+new_w]
        patch_target_img = full_target_img[:,top:top+new_h, left:left+new_w]
        patch_input_imgs.append(patch_input_img)
        patch_target_imgs.append(patch_target_img)
    return torch.stack(patch_input_imgs,dim=0), torch.stack(patch_target_imgs,dim=0)