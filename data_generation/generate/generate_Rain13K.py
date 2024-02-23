import argparse
import multiprocessing
import os
import pathlib
import sys

import cv2
import lmdb
from torch.utils.data import DataLoader

from generate.img_to_token import img_to_token

parent_path = pathlib.Path(__file__).absolute().parent.parent
parent_path = os.path.abspath(parent_path)
sys.path.append(parent_path)
os.chdir(parent_path)
print(f'>-------------> parent path {parent_path}')
print(f'>-------------> current work dir {os.getcwd()}')

import numpy as np

from PIL import Image
from os.path import join

import torch
from torchvision.datasets import VisionDataset

from vqgan.load import encode_transform

CPU_COUNT = multiprocessing.cpu_count()


class LMDBDataset(VisionDataset):
    def __init__(
            self,
            root: str,
            target_root,
            transform=None,
            target_transform=None,
            transforms=None,
            transform_name=None
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self.target_root = target_root
        
        self.img_db = lmdb.open(root).begin()
        self.target_db = lmdb.open(target_root).begin()
        
        with open(join(root, 'meta_info.txt'), 'rb') as f:
            file_list = f.readlines()
        
        ids = [i.decode().split(' ')[0].split('.')[0] for i in file_list]
        self.ids = list(sorted(ids))
        
        self.transform_name = transform_name
    
    def _load_image(self, id):
        img_byte = self.img_db.get(id.encode())
        image_buf = np.frombuffer(img_byte, dtype=np.uint8)
        img = cv2.imdecode(image_buf, cv2.IMREAD_COLOR)
        image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        return image.convert("RGB")
    
    def _load_target(self, id):
        img_byte = self.target_db.get(id.encode())
        image_buf = np.frombuffer(img_byte, dtype=np.uint8)
        img = cv2.imdecode(image_buf, cv2.IMREAD_COLOR)
        image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        return image.convert("RGB")
    
    def __getitem__(self, index: int):
        id = self.ids[index]
        image = self._load_image(id)
        target_img = self._load_target(id)
        
        images = self.transform(image)
        target_imgs = self.transform(target_img)
        
        data_list = []
        if self.transform_name == 'six_crop_encode_transform':
            for _img, _target_img in zip(images, target_imgs):
                _data = torch.stack([_img, _target_img], dim=0)
                data_list.append(_data)
        else:
            _data = torch.stack([images, target_imgs], dim=0)
            data_list.append(_data)
        
        data = torch.cat(data_list, dim=0)
        
        return data
    
    def __len__(self) -> int:
        return len(self.ids)


def convert_img_to_token(args, device=None):
    dataset = LMDBDataset(args.input_data, args.target_data, transform=encode_transform,
                          transform_name='encode_transform')
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_work)
    img_to_token(args, data_loader, args.output_path, device=device)


def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--input_data", type=str, default="Rain13K_lmdb/input.lmdb")
    parser.add_argument("--target_data", type=str, default="Rain13K_lmdb/target.lmdb")
    parser.add_argument("--output_path", type=str, default="vq_token/Rain13K")
    
    parser.add_argument("--num_work", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--dp_mode", action='store_true', default=False)
    parser.add_argument("--model_name_or_path", type=str, default="weight/vqgan-f16-8192-laion")
    args = parser.parse_args()
    
    return args


if __name__ == '__main__':
    args = get_args()
    
    # input_root = '/home/ma-user/work/data/tmp_data/Rain13K_lmdb/input.lmdb'
    # target_root = '/home/ma-user/work/data/tmp_data/Rain13K_lmdb/target.lmdb'
    # out_root = '/home/ma-user/work/data/vq_token/Rain13K'
    
    device = f'cuda:{0}'
    convert_img_to_token(args, device=device)
