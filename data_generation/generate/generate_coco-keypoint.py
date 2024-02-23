import os
import pathlib
import sys

parent_path = pathlib.Path(__file__).absolute().parent.parent
parent_path = os.path.abspath(parent_path)
sys.path.append(parent_path)
os.chdir(parent_path)
print(f'>-------------> parent path {parent_path}')
print(f'>-------------> current work dir {os.getcwd()}')

import argparse
import glob
import multiprocessing

from PIL import Image
from os.path import join

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset

from vqgan.load import encode_transform
from generate.img_to_token import img_to_token

CPU_COUNT = multiprocessing.cpu_count()


class KeyPointDataset(VisionDataset):
    def __init__(
            self,
            root: str,
            target_root,
            transform=None,
            target_transform=None,
            transforms=None,
            transform_name='encode_transform'
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self.target_root = target_root
        
        file_list = glob.glob(join(root, '*.jpg'))
        ids = [os.path.basename(i).split('.')[0] for i in file_list]
        self.ids = list(sorted(ids))
        self.transform_name = transform_name
    
    def _load_image(self, id: int):
        path = join(self.root, f'{id}.jpg')
        return Image.open(path).convert("RGB")
    
    def _load_target(self, id: int):
        path = join(self.target_root, f'{id}.jpg')
        return Image.open(path).convert("RGB")
    
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
    dataset = KeyPointDataset(args.input_data, args.target_data, transform=encode_transform)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_work)
    img_to_token(args, data_loader, args.output_path, device=device)


def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--input_data", type=str, default="coco-pose/GT/val2017/visual-crop/images")
    parser.add_argument("--target_data", type=str, default="coco-pose/GT/val2017/visual-crop/keypoints")
    parser.add_argument("--output_path", type=str, default="vq_token/coco-crop/val2017")
    
    parser.add_argument("--num_work", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--dp_mode", action='store_true', default=False)
    parser.add_argument("--model_name_or_path", type=str, default="weight/vqgan-f16-8192-laion")
    args = parser.parse_args()
    
    return args


if __name__ == '__main__':
    args = get_args()
    
    device = f'cuda:{0}'
    convert_img_to_token(args, device=device)
