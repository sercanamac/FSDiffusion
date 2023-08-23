"""
    adopted from
        - https://github.com/shubhtuls/PixelTransformer/blob/03b65b8612fe583b3e35fc82b446b5503dd7b6bd/data/shapenet.py
        - https://github.com/hzxie/Pix2Vox
    
"""

import os
import glob
import json
import socket

import scipy.io
import h5py
import numpy as np
from PIL import Image
from termcolor import colored, cprint

import torch
import torch.nn.functional as F
import torchvision.utils as vutils
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import InterpolationMode

from datasets.base_dataset import BaseDataset
from utils.pix3d_util import downsample_voxel

hostname = socket.gethostname()

class RandomNoise(object):
    def __init__(self,
                 noise_std,
                 eigvals=(0.2175, 0.0188, 0.0045),
                 eigvecs=((-0.5675, 0.7192, 0.4009), (-0.5808, -0.0045, -0.8140), (-0.5836, -0.6948, 0.4203))):
        self.noise_std = noise_std
        self.eigvals = np.array(eigvals)
        self.eigvecs = np.array(eigvecs)

    def __call__(self, rendering_images):
        alpha = np.random.normal(loc=0, scale=self.noise_std, size=3)
        noise_rgb = \
            np.sum(
                np.multiply(
                    np.multiply(
                        self.eigvecs,
                        np.tile(alpha, (3, 1))
                    ),
                    np.tile(self.eigvals, (3, 1))
                ),
                axis=1
            )

        # Allocate new space for storing processed images
        c, h, w = rendering_images.shape
        processed_images = torch.zeros_like(rendering_images)
        for i in range(c):
            processed_images[i, :, :] = rendering_images[i, :, :]
            processed_images[i, :, :] += noise_rgb[i]
        
        return processed_images

class RandomPermuteRGB(object):
    def __call__(self, rendering_images):
        # assert (isinstance(rendering_images, np.ndarray))

        random_permutation = np.random.permutation(3)
        rendering_images = rendering_images[random_permutation, ...] 

        return rendering_images

class ShapenetImg2ShapeDataset(BaseDataset):
    def initialize(self, opt, phase='train', cat='chair', input_txt=None, by_imgs=True):
        self.opt = opt
        self.max_dataset_size = opt.max_dataset_size

        dataroot = opt.dataroot
        # with open(f'{dataroot}/ShapeNet/info.json') as f:
        with open(f'dataset_info_files/info-shapenet.json') as f:
            self.info = json.load(f)
            
        self.cat_to_id = self.info['cats']
        self.id_to_cat = {v: k for k, v in self.cat_to_id.items()}
        
        if cat == 'all':
            all_cats = self.info['all_cats']
        else:
            all_cats = [cat]

        self.model_list = []
        self.cats_list = []
        self.img_paths = []
        self.mask_paths = []
        for c in all_cats:
            synset = self.info['cats'][c]
            # with open(f'{dataroot}/ShapeNet/filelists/{synset}_{phase}.lst') as f:
            with open(f'dataset_info_files/ShapeNet_filelists/{synset}_{phase}.lst') as f:
                model_list_s = []
                for l in f.readlines():
                    model_id = l.rstrip('\n')
                    path = f'{dataroot}/ShapeNet/SDF_v1_64/{synset}/{model_id}/ori_sample.h5'
                    img_path =f"/home/amac/data/ShapeNet55_3DOF-VC_LRBg/{synset}/{model_id}/image_output"
                    mask_path = f"/home/amac/data/ShapeNet55_3DOF-VC_LRBg/{synset}/{model_id}/segmentation"
                    if os.path.exists(path):
                        model_list_s.append(path)
                    if os.path.exists(img_path):
                        self.img_paths.append(img_path)
                    if os.path.exists(mask_path):
                        self.mask_paths.append(mask_path)

                        

                self.model_list += model_list_s
                self.cats_list += [synset] * len(model_list_s)
                print('[*] %d samples for %s (%s).' % (len(model_list_s), self.id_to_cat[synset], synset))


        self.N = len(self.img_list)
        self.to_tensor = transforms.ToTensor()

        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        if phase == 'train':
            self.transforms = transforms.Compose([
                transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25),
                RandomNoise(0.1),
                transforms.RandomAffine(0, scale=(0.7, 1.25), interpolation=InterpolationMode.BILINEAR),
                transforms.Normalize(mean, std),
                transforms.RandomHorizontalFlip(),
                RandomPermuteRGB(),
                transforms.Resize((256, 256)),
            ])
        else:
            self.transforms = transforms.Compose([
                transforms.Normalize(mean, std),
                transforms.Resize((256, 256)),
            ])

        self.n_view = 1
    
    def process_img(self, img):
        img_t = self.to_tensor(img)

        _, oh, ow = img_t.shape

        ls = max(oh, ow)

        pad_h1, pad_h2 = (ls - oh) // 2, (ls - oh) - (ls - oh) // 2
        pad_w1, pad_w2 = (ls - ow) // 2, (ls - ow) - (ls - ow) // 2
        img_t = F.pad(img_t[None, ...], (pad_w1, pad_w2, pad_h1, pad_h2), mode='constant', value=0)

        img_t = self.transforms(img_t[0])

        return img_t

    def read_vox(self, f):
        gt_size = 32

        voxel_p = f
        voxel = scipy.io.loadmat(voxel_p)['voxel']

        # downsample
        voxel = downsample_voxel(voxel, 0.5, (gt_size, gt_size, gt_size))
        voxel = torch.from_numpy(voxel)
        voxel = voxel.float()
        return voxel

    def __getitem__(self, index):
        
        cat_name = self.cats_list[index]
        sdf_h5_file = self.model_list[index]

        h5_f = h5py.File(sdf_h5_file, 'r')
        sdf = h5_f['pc_sdf_sample'][:].astype(np.float32)
        sdf = torch.Tensor(sdf).view(1, 64, 64, 64)

        thres = self.opt.trunc_thres
        if thres != 0.0:
            sdf = torch.clamp(sdf, min=-thres, max=thres)

        # load img; randomly sample 1
        imgs = []
        img_paths = []
        imgs_all_view = self.img_list[index]
        sample_ixs = np.random.choice(len(imgs_all_view), self.n_view)
        for ix in sample_ixs:
            p = imgs_all_view[ix]
            im = Image.open(p).convert('RGB')
            im = self.process_img(im)
            imgs.append(im)
            img_paths.append(p)

        imgs = torch.stack(imgs).clamp(-1., 1.)
        img = imgs[0]
        img_path = img_paths[0]

        gt_vox_path = self.gt_voxel_list[index]
        gt_vox = self.read_vox(gt_vox_path) # already downsample

        ret = {
            'sdf': sdf, 'sdf_path': sdf_h5_file,
            'img': img, 'img_path': img_path,
            'imgs': imgs, 'img_paths': img_paths,
            'gt_vox': gt_vox, 'gt_vox_path': gt_vox_path,
            'cat_str': cat_name,
        }

        return ret

    def __len__(self):
        return self.N

    def name(self):
        return 'ShapenetImg2ShapeDataset'