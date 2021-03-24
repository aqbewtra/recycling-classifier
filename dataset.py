from glob import glob
import os
from PIL import Image

from torch.utils.data import Dataset

import matplotlib.pyplot as plt 

from torchvision import transforms

import torch

class RecyclingDataset(Dataset):
    def __init__(self, glass_path, metal_path, misc_plastic_path, paper_path, plastic_path):
        
        self.glass_paths = glob(os.path.join(glass_path, '*.jpg'))
        self.metal_paths = glob(os.path.join(metal_path, '*.jpg'))
        self.misc_plastic_paths = glob(os.path.join(misc_plastic_path, '*.jpg'))
        self.paper_paths = glob(os.path.join(paper_path, '*.jpg'))
        self.plastic_paths = glob(os.path.join(plastic_path, '*.jpg'))

        self.path_dict = dict()

        #self.bottle_label = torch.tensor([0])
        #self.can_label = torch.tensor([1])
        self.glass_label = torch.tensor([1,0,0,0,0])
        self.metal_label = torch.tensor([0,1,0,0,0])
        self.misc_plastic_label = torch.tensor([0,0,1,0,0])
        self.paper_label = torch.tensor([0,0,0,1,0])
        self.plastic_label = torch.tensor([0,0,0,0,1])

        for i, i_path in enumerate(self.glass_paths):
            self.path_dict.update({i: (i_path, self.glass_label)})
        
        add = len(self.glass_paths) - 1

        for j, j_path in enumerate(self.metal_paths):
            self.path_dict.update({(j + add): (j_path, self.metal_label)})

        add += len(self.metal_paths) - 1

        for k, k_path in enumerate(self.misc_plastic_paths):
            self.path_dict.update({(k + add): (k_path, self.misc_plastic_label)})

        add += len(self.misc_plastic_paths) - 1

        for l, l_path in enumerate(self.paper_paths):
            self.path_dict.update({(l + add): (l_path, self.paper_label)})

        add += len(self.paper_paths) - 1

        for m, m_path in enumerate(self.plastic_paths):
            self.path_dict.update({(m + add): (m_path, self.plastic_label)})
        
        self.im_to_tensor = transforms.ToTensor()
        self.fix_size = transforms.Compose([transforms.Resize((256,256))])
        #TRANSFORMS HERE
    
    def __len__(self):
        return len(self.path_dict)

    def __getitem__(self, index):
        im_path, label = self.path_dict[index]

        img = Image.open(im_path).convert('RGB')
        img = self.im_to_tensor(img)
        img = self.fix_size(img)

        return img, label

        