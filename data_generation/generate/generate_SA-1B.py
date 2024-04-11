import os
import pathlib
import sys

from torch.utils.data import DataLoader

parent_path = pathlib.Path(__file__).absolute().parent.parent
parent_path = os.path.abspath(parent_path)
sys.path.append(parent_path)
os.chdir(parent_path)
print(f'>-------------> parent path {parent_path}')
print(f'>-------------> current work dir {os.getcwd()}')

import glob
import json
import argparse
import subprocess
import multiprocessing
import numpy as np

from tqdm import tqdm
from PIL import Image
from os.path import join
from joblib import delayed, Parallel
from pycocotools import mask as mask_utils

import torch
from torchvision.datasets import VisionDataset

from generate.img_to_token import img_to_token
from vqgan.load import six_crop_encode_transform

CPU_COUNT = multiprocessing.cpu_count()


def convert_anns_to_mask(sam_label):
    # device = f'cuda:{0}'
    device = f'cpu'
    
    image_info = sam_label['image']
    anns = sam_label['annotations']
    width, height, file_name = image_info['width'], image_info['height'], image_info['file_name']
    
    if len(anns) == 0:
        return
    
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    mask_img = torch.zeros((height, width, 3), device=device)
    one_img = torch.ones((height, width, 3), device=device)
    
    for ann in sorted_anns:
        mask = mask_utils.decode(ann['segmentation'])
        mask = torch.tensor(mask, device=device)
        mask = torch.repeat_interleave(mask.unsqueeze(dim=2), repeats=3, dim=2)
        
        color_mask = torch.rand(3, device=device) * one_img
        mask_img += color_mask * mask
        
        del mask, color_mask
        # torch.cuda.empty_cache()
    
    mask_img_npy = mask_img.cpu().numpy()
    mask_img_npy = (255 * mask_img_npy).astype(np.uint8)
    
    del mask_img, one_img
    # torch.cuda.empty_cache()
    
    return mask_img_npy


def convert_sam_label(json_dir, out_dir, tar_name):
    print(f'>----------------------: convert sam label: {tar_name} ...\n')
    
    os.makedirs(out_dir, exist_ok=True)
    
    def _convert(_json_path, _out_dir):
        data_name = os.path.basename(_json_path)
        out_path = join(_out_dir, data_name.replace('json', 'png'))
        with open(_json_path) as f:
            sam_label = json.load(f)
        
        mask_img = convert_anns_to_mask(sam_label)
        mask_img = Image.fromarray(mask_img)
        mask_img.save(out_path)
    
    json_list = glob.glob(join(json_dir, '*.json'))
    
    Parallel(n_jobs=CPU_COUNT)(
            delayed(_convert)(index, json_path, out_dir) for json_path in tqdm(json_list))


class SamDataset(VisionDataset):
    def __init__(
            self,
            root: str,
            mask_root,
            transform=None,
            target_transform=None,
            transforms=None
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self.mask_root = mask_root
        
        file_list = glob.glob(join(root, '*.jpg'))
        ids = [os.path.basename(i).split('.')[0] for i in file_list]
        self.ids = list(sorted(ids))
    
    def _load_image(self, id: int):
        path = join(self.root, f'{id}.jpg')
        return Image.open(path).convert("RGB")
    
    def _load_mask(self, id: int):
        path = join(self.mask_root, f'{id}.png')
        return Image.open(path).convert("RGB")
    
    def __getitem__(self, index: int):
        id = self.ids[index]
        image = self._load_image(id)
        mask_img = self._load_mask(id)
        
        images = self.transform(image)
        mask_imgs = self.transform(mask_img)
        
        data_list = []
        for _img, _mask_img in zip(images, mask_imgs):
            _data = torch.stack([_img, _mask_img], dim=0)
            data_list.append(_data)
        
        data = torch.cat(data_list, dim=0)
        
        return data
    
    def __len__(self) -> int:
        return len(self.ids)


def convert_img_to_token(args, img_data_dir, mask_data_dir, out_dir, tar_name, device=None):
    print(f'>----------------------: Convert img to token: {tar_name} ...')
    
    dataset = SamDataset(img_data_dir, mask_data_dir, transform=six_crop_encode_transform([800, 800]))
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_work)
    img_to_token(args, data_loader, out_dir, device=device)


def unzip_data(tar_root, img_json_dir, tar_name):
    print(f'>----------------------: Unzip data: {tar_name} ...')
    
    local_tar_path = join(tar_root, f'{tar_name}.tar')
    os.makedirs(img_json_dir, exist_ok=True)
    
    cmd = f'tar -xf {local_tar_path} -C {img_json_dir}'
    subprocess.check_call(args=cmd, shell=True)


def remove_tmpfile(img_json_dir, mask_dir, tar_name):
    print(f'>----------------------: Remove tmpfile: {tar_name} ...')
    
    tmp_files = [
            img_json_dir,
            mask_dir
    ]
    
    for tmp_file in tmp_files:
        cmd = f'rm -rf {tmp_file}'
        subprocess.check_call(args=cmd, shell=True)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tar_root", type=str, default="data/SA-1B/tar")
    parser.add_argument("--img_json_root", type=str, default="data/SA-1B/tmp/img_json")
    parser.add_argument("--mask_root", type=str, default="data/SA-1B/tmp/mask")
    parser.add_argument("--output_path", type=str, default="vq_token/SA-1B")
    
    parser.add_argument("--num_work", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--dp_mode", action='store_true', default=False)
    parser.add_argument("--model_name_or_path", type=str, default="weight/vqgan-f16-8192-laion")
    args = parser.parse_args()
    
    return args


if __name__ == '__main__':
    args = get_args()
    
    exclusion_data = []
    
    tar_name_list = os.listdir(args.tar_root)
    tar_name_list = [i.split('.')[0] for i in tar_name_list if i[-3:] == 'tar']
    
    if os.path.exists(args.output_path):
        exist_token_name_list = os.listdir(args.output_path)
    else:
        exist_token_name_list = []
    
    tar_name_list = list(set(tar_name_list) - set(exist_token_name_list))
    tar_name_list = sorted(tar_name_list)
    
    for index, tar_name in enumerate(tar_name_list):
        
        if tar_name in exclusion_data:
            continue
        
        print(f'\n\nProcessing sam data: {tar_name}  {index + 1}/{len(tar_name_list)} ...')
        img_json_dir = join(args.img_json_root, tar_name)
        mask_dir = join(args.mask_root, tar_name)
        out_dir = join(args.output_path, tar_name)
        
        unzip_data(args.tar_root, img_json_dir, tar_name)
        convert_sam_label(img_json_dir, mask_dir, tar_name)
        
        device = 'cuda'
        convert_img_to_token(args, img_json_dir, mask_dir, out_dir, tar_name, device=device)
        
        remove_tmpfile(img_json_dir, mask_dir, tar_name)
