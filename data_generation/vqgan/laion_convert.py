import argparse
import json
import os
import pathlib
import random
import subprocess
import tarfile
import time
from multiprocessing import Pool

import numpy as np
import mindspore
import mindspore.nn as nn
from timm.data import ImageDataset
from mindspore.utils.data import DataLoader

mindspore.set_grad_enabled(False)
random.seed(42)

from load import encode_transform, load_model

# set seed
random_seed = 1
mindspore.manual_seed(random_seed)
mindspore.cuda.manual_seed(random_seed)
mindspore.backends.cudnn.deterministic = True
mindspore.backends.cudnn.benchmark = False
np.random.seed(random_seed)

# args
parser = argparse.ArgumentParser()
# unzip
parser.add_argument("--folder_path", type=str, default='/cache/data/laion400m-images/part0/')
parser.add_argument("--extract_path", type=str, default='/home/ma-user/work/laion400m-images/part0_jpg/')
parser.add_argument("--prefix", type=str, default='laion_part0')
parser.add_argument("--num_processes", type=int, default=10)
# vqgan convert
parser.add_argument("--data", type=str, default='/cache/laion_jpg/part0/')  # folder of unziped imgs
parser.add_argument("--batch_size", type=str, default=256)  # folder of imgs
parser.add_argument("--type", type=str, default="internlm")
parser.add_argument("--output", type=str, default="/cache/laion_train_convert/part0/")
parser.add_argument("--model_name_or_path", type=str, default="/cache/ckpt/vqgan-f16-8192-laion")
args = parser.parse_args()
args.data = args.extract_path

# unzip part
unzip_start_time = time.time()


def extract_tar(tar_info):
    tar_file, folder_path, extract_path, prefix = tar_info
    tar_file_path = os.path.join(folder_path, tar_file)
    target_folder = os.path.join(extract_path, f"{prefix}_{tar_file[:-4]}")
    
    with tarfile.open(tar_file_path, 'r') as tar:
        tar.extractall(target_folder)
    
    print(f"Extracted {tar_file} to {target_folder}")
    
    cmd_txt = 'yes | rm -r ' + target_folder + '/*.txt'
    subprocess.run(cmd_txt, shell=True)
    cmd_json = 'yes | rm -r ' + target_folder + '/*.json'
    subprocess.run(cmd_json, shell=True)
    cmd_tar_file = 'yes | rm -r ' + tar_file_path
    subprocess.run(cmd_tar_file, shell=True)


def extract_all_tarfiles_parallel(folder_path, extract_path, prefix, num_processes=4):
    file_list = [file for file in os.listdir(folder_path) if file.endswith('.tar')]
    tar_info_list = [(tar_file, folder_path, extract_path, prefix) for tar_file in file_list]
    
    with Pool(num_processes) as pool:
        pool.map(extract_tar, tar_info_list)


extract_all_tarfiles_parallel(args.folder_path, args.extract_path, args.prefix, args.num_processes)
print('########### unzip time: ', time.time() - unzip_start_time)

convert_start_time = time.time()


# convert part
def list_subdir(folder_path):
    subdir = [f.name for f in os.scandir(folder_path) if f.is_dir()]
    return subdir


dir_names = list_subdir(args.data)

print('Strating convert via vqgan...')
print(args)
print(len(dir_names))

vq_model = load_model(args.model_name_or_path)
vq_model = vq_model.cuda().eval()


class ParallelWrapper(nn.Module):
    def __init__(self, vq_model, func='encode'):
        super(ParallelWrapper, self).__init__()
        self.vq_model = vq_model
        self.func = func
    
    def forward(self, x):
        return getattr(self.vq_model, self.func)(x)


encoder = ParallelWrapper(vq_model)
encoder = nn.DataParallel(encoder)


def dumps(data):
    seqlen = len(data)
    saved_bin = str.encode(json.dumps(dict(tokens=data)) + "\n")
    return {"bin": saved_bin, "length": seqlen}


for idx, sub_dir_name in enumerate(dir_names):
    if idx % 10 == 0:
        print(idx)
    
    output_dir = os.path.join(args.output, sub_dir_name)
    
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    out_bin = os.path.join(output_dir, "train.bin")
    out_meta = os.path.join(output_dir, "train.bin.meta")
    
    pathlib.Path(out_bin).touch(exist_ok=True)
    pathlib.Path(out_meta).touch(exist_ok=True)
    
    dataset = ImageDataset(os.path.join(args.data, sub_dir_name), transform=encode_transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    
    from tqdm import tqdm
    
    cu = 0
    cu_seqlens = []
    with open(out_bin, "wb+") as bin_file:
        for i, (imgs, _) in enumerate(tqdm(loader)):
            imgs = imgs.cuda()
            quantized_states, indices = encoder(imgs)
            
            for indices_i in indices.tolist():
                token = dumps(indices_i)
                seqlen = token["length"]  # 256
                token_data = token["bin"]
                bin_file.write(token_data)
                # print((cu, seqlen))
                cu_seqlens.append((cu, seqlen))
                cu += len(token_data)
    cu_seqlens = np.array(cu_seqlens, dtype=np.int64)
    with open(out_meta, "wb+") as meta_file:
        np.save(meta_file, cu_seqlens)

print('########### unzip time: ', time.time() - convert_start_time)
print('Finish convert...')
