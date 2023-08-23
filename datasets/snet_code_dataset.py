"""
    adopted from: https://github.com/shubhtuls/PixelTransformer/blob/03b65b8612fe583b3e35fc82b446b5503dd7b6bd/data/shapenet.py
"""
import os.path
import json

import h5py
import numpy as np
from termcolor import colored, cprint

import torch
import torch.nn.functional as F
import torchvision.utils as vutils
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import InterpolationMode
from utils.util_3d import sdf_to_mesh
from pytorch3d.ops import sample_points_from_meshes
from datasets.base_dataset import BaseDataset
from utils.demo_util import preprocess_image
import glob


# from https://github.com/laughtervv/DISN/blob/master/preprocessing/info.json
class ShapeNetImg2ShapeDataset(BaseDataset):

    def initialize(self, opt, phase='train', cat='all', res=64):
        self.opt = opt
        self.max_dataset_size = opt.max_dataset_size
        self.res = res
        self.few = opt.few
        dataroot = "/home/amac/SDFusion/data"
        # with open(f'{dataroot}/ShapeNet/info.json') as f:
        with open(f'dataset_info_files/info-shapenet.json') as f:
            self.info = json.load(f)
            
        self.cat_to_id = self.info['cats']
        self.id_to_cat = {v: k for k, v in self.cat_to_id.items()}
        
        if cat == 'all':
            all_cats = self.info[phase + "_cats"]
        else:
            all_cats = [cat]

        all_imgs = glob.glob("/home/amac/data/ShapeNet55_3DOF-VC_LRBg/*/*/image_output/*.png")

        self.model_list = []
        self.cats_list = []
        model_ids = []
        self.cat2model_list = {}
        with open("data_split.json", "r") as f:
            splo = json.load(f)
        splo = splo[phase]
        self.splo = splo
        for c in all_cats:
            synset = self.info['cats'][c]
            # with open(f'{dataroot}/ShapeNet/filelists/{synset}_{phase}.lst') as f:
            model_list_s = []
            for l in splo[synset]:
                model_id = l.rstrip('\n')
                if res == 64:

                    path = f'{dataroot}/ShapeNet/SDF_v1_64/{synset}/{model_id}/ori_sample.h5'
                else:

                    path = f'{dataroot}/ShapeNet/SDF_v2/resolution_{self.res}/{synset}/{model_id}/ori_sample_grid.h5'
                
                if os.path.exists(path):
                    model_list_s.append(path)
                model_ids.append(model_id)
                if model_id not in self.cat2model_list:

                    self.cat2model_list[synset] = [path]
                else:
                    self.cat2model_list[synset].append(path)


            self.model_list += model_list_s
            self.cats_list += [synset] * len(model_list_s)
            print('[*] %d samples for %s (%s).' % (len(model_list_s), self.id_to_cat[synset], synset))
        
        self.model2views = {model_id: glob.glob(f"/home/amac/data/ShapeNet55_3DOF-VC_LRBg/*/{model_id}/image_output/*.png") for model_id in model_ids}
        np.random.default_rng(seed=0).shuffle(self.model_list)
        np.random.default_rng(seed=0).shuffle(self.cats_list)
        self.model_list = self.model_list[:self.max_dataset_size]
        cprint('[*] %d samples loaded.' % (len(self.model_list)), 'yellow')

        self.N = len(self.model_list)
        
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.Resize((256, 256)),
        ])
    def __getitem__(self, index):
        try:
                
            synset = self.cats_list[index]
            sdf_h5_file = self.model_list[index]
            assert synset == sdf_h5_file.split("/")[-3]
            h5_f = h5py.File(sdf_h5_file, 'r')
            sdf = h5_f['pc_sdf_sample'][:].astype(np.float32)

            sdf = torch.Tensor(sdf).view(1, self.res, self.res, self.res)
            # print(sdf.shape)
            # sdf = sdf[:, :64, :64, :64]

            thres = self.opt.trunc_thres
            if thres != 0.0:
                sdf = torch.clamp(sdf, min=-thres, max=thres)
            z = np.load(sdf_h5_file.replace("ori_sample.h5", "latent_code.npy"), allow_pickle=True).squeeze(0)
            view = np.random.choice(self.model2views[sdf_h5_file.split("/")[-2]], 1)[0]
            
            _, img = preprocess_image(str(view), str(view).replace("image_output", "segmentation"))
            img = self.transforms(img)
           
            ret = {
                'sdf': sdf,
                'z': z,
                'img': img,
                'cat_id': synset,
                'cat_str': self.id_to_cat[synset],
                'path': sdf_h5_file,
            }
            if self.few:
                listo = self.cat2model_list[synset]
                sup_path = np.random.choice(listo, 1)[0]
                sup_code = np.load(sup_path.replace("ori_sample.h5", "latent_code.npy")).squeeze()
                ret["sup_z"] = sup_code
        except Exception as e:
            print(e)
            return self.__getitem__(np.random.randint(low=0, high=self.N))

        return ret

    def __len__(self):
        return self.N

    def name(self):
        return 'ShapeNetImg2ShapeDataset'