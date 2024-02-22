import os
import pathlib
import sys

parent_path = pathlib.Path(__file__).absolute().parent.parent
parent_path = os.path.abspath(parent_path)
sys.path.append(parent_path)
os.chdir(parent_path)
# print(f'>-------------> parent path {parent_path}')
# print(f'>-------------> current work dir {os.getcwd()}')

import argparse
import glob
import jsonlines

from os.path import join
from tqdm import tqdm
from PIL import Image
# from joblib import Parallel, delayed

import mindspore
from timm.data import ImageDataset
from mindspore.utils.data import DataLoader

from vqgan.load import encode_transform
from vqgan.utils import init_vqgan_encoder, get_multiprocess
from generate.img_to_token import data_loader_to_token, save_bin_and_meta_file


class ImageDatasetNoLabel(ImageDataset):
    def __getitem__(self, index):
        img, target = self.parser[index]
        img = img.read() if self.load_bytes else Image.open(img).convert('RGB')
        self._consecutive_errors = 0
        if self.transform is not None:
            img = self.transform(img)
        return img


def convert_img_to_token(args, data_dir, out_dir, encoder, device=None):
    all_data_bin_list, all_cu_seq_len_list = [], []
    
    for sub_dir_name in os.listdir(data_dir):
        input_dir = os.path.join(data_dir, sub_dir_name)
        
        if not os.path.exists(input_dir) or len(glob.glob(join(input_dir, '*'))) == 0:
            # print('Path not exist: ', input_dir)
            continue
        
        dataset = ImageDatasetNoLabel(input_dir, transform=encode_transform)
        new_multiprocess_ctx = get_multiprocess()
        data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=8,
                                 multiprocessing_context=new_multiprocess_ctx)
        
        data_bin_list, cu_seq_len_list = data_loader_to_token(encoder, data_loader, device)
        all_data_bin_list.extend(data_bin_list)
        all_cu_seq_len_list.extend(cu_seq_len_list)
    
    save_bin_and_meta_file(out_dir, all_data_bin_list, all_cu_seq_len_list)


def convert_single_gpu(video_name_list, num_work, index):
    work_len = len(video_name_list) // num_work
    start, end = index * work_len, (index + 1) * work_len
    if index == num_work - 1:
        work_video_name_list = video_name_list[start:]
    else:
        work_video_name_list = video_name_list[start: end]
    
    device = f'cuda:{index}'
    encoder = init_vqgan_encoder(args.model_name_or_path, device)
    
    for video_name in tqdm(work_video_name_list):
        data_dir = join(args.data_root, video_name)
        out_dir = join(args.output_root, video_name)
        
        convert_img_to_token(args, data_dir, out_dir, encoder, device)


def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--video_info_json", type=str,
                        default="hdvila_100m/cut_video_results/cut_part0.jsonl")
    parser.add_argument("--data_root", type=str, default='hdvila_100m/video_clips_imgs')
    parser.add_argument("--output_root", type=str, default="vq_token/hdvila_100m_2")
    
    parser.add_argument("--num_gpu", type=int, default=-1)
    parser.add_argument("--batch_size", type=str, default=128)
    parser.add_argument("--model_name_or_path", type=str, default="weight/vqgan-f16-8192-laion")
    args = parser.parse_args()
    
    if args.num_gpu == -1:
        args.num_gpu = mindspore.cuda.device_count()
    
    return args


if __name__ == '__main__':
    args = get_args()
    num_gpu = args.num_gpu
    
    print('Convert video info json: ', args.video_info_json)
    
    with jsonlines.open(args.video_info_json, 'r') as f:
        video_name_list = [l.split('.')[0] for l in f]
    
    if os.path.exists(args.output_root):
        exist_token_name = set(os.listdir(args.output_root))
    else:
        exist_token_name = []
    
    video_name_list = list(set(video_name_list) - set(exist_token_name))
    
    for i in range(num_gpu):
        convert_single_gpu(video_name_list, num_gpu, i)
    
    # Parallel(n_jobs=num_gpu)(delayed(convert_single_gpu)(video_name_list, num_gpu, i) for i in range(num_gpu))
