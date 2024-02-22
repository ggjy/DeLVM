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
import subprocess
import tarfile
import time
from multiprocessing import Pool

from timm.data import ImageDataset
from mindspore.utils.data import DataLoader
from tqdm import tqdm

from generate.img_to_token import img_to_token
from vqgan.load import encode_transform


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


def list_subdir(folder_path):
    subdir = [f.name for f in os.scandir(folder_path) if f.is_dir()]
    return subdir


def get_args():
    parser = argparse.ArgumentParser()
    
    # unzip
    parser.add_argument("--folder_path", type=str, default='/cache/data/laion400m-images/part0/')
    parser.add_argument("--extract_path", type=str, default='/home/ma-user/work/laion400m-images/part0_jpg/')
    parser.add_argument("--prefix", type=str, default='laion_part0')
    parser.add_argument("--num_processes", type=int, default=10)
    # vqgan convert
    parser.add_argument("--data", type=str, default='/cache/laion_jpg/part0/')  # folder of unziped imgs
    parser.add_argument("--batch_size", type=str, default=256)  # folder of imgs
    parser.add_argument("--output", type=str, default="/cache/laion_train_convert/part0/")
    parser.add_argument("--model_name_or_path", type=str, default="weight/vqgan-f16-8192-laion")
    args = parser.parse_args()
    args.data = args.extract_path
    
    return args


if __name__ == '__main__':
    args = get_args()
    
    unzip_start_time = time.time()
    extract_all_tarfiles_parallel(args.folder_path, args.extract_path, args.prefix, args.num_processes)
    print('########### unzip time: ', time.time() - unzip_start_time)
    
    dir_names = list_subdir(args.data)
    
    device = 'cuda:0'
    
    for idx, sub_dir_name in enumerate(tqdm(dir_names)):
        output_dir = os.path.join(args.output, sub_dir_name)
        
        dataset = ImageDataset(args.input_data, args.target_data, transform=encode_transform)
        data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_work)
        img_to_token(args, data_loader, args.output_path, device=device)
    
    print('Finish convert...')
