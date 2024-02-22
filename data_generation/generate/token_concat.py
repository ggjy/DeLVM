import json
import os
from itertools import chain
from os.path import join

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm


def img_concat(data_root, output_root, data_name, token_num_per_sentence):
    data_dir = join(data_root, data_name)
    out_dir = join(output_root, data_name)
    
    os.makedirs(out_dir, exist_ok=True)
    
    data_bin_path = os.path.join(data_dir, "train.bin")
    out_data_bin_path = os.path.join(out_dir, "train.bin")
    out_data_meta_path = os.path.join(out_dir, "train.bin.meta")
    
    with open(data_bin_path, "r") as bin_file:
        data_bin = bin_file.readlines()
    
    cu = 0
    new_data_bin = []
    cu_seq_len_list = []
    
    sentence = []
    for index, data in enumerate(data_bin):
        data = json.loads(data)['tokens']
        if index > 0 and index % token_num_per_sentence == 0:
            tokens = list(chain(*sentence))
            seq_len = len(tokens)
            saved_bin = str.encode(json.dumps(dict(tokens=tokens)) + "\n")
            
            new_data_bin.append(saved_bin)
            cu_seq_len_list.append((cu, seq_len))
            cu += len(saved_bin)
            sentence = []
        
        sentence.append(data)
    
    tokens = list(chain(*sentence))
    seq_len = len(tokens)
    saved_bin = str.encode(json.dumps(dict(tokens=tokens)) + "\n")
    
    new_data_bin.append(saved_bin)
    cu_seq_len_list.append((cu, seq_len))
    cu += len(saved_bin)
    
    with open(out_data_bin_path, "wb+") as out_bin_file:
        out_bin_file.writelines(new_data_bin)
    np.save(out_data_meta_path, cu_seq_len_list)
    os.rename(f'{out_data_meta_path}.npy', out_data_meta_path)


if __name__ == '__main__':
    token_num_per_sentence = 6
    file_name = 'Rain13K'
    data_root = '/home/ma-user/work/data/vq_token'
    
    data_dir = join(data_root, file_name)
    output_dir = join(data_root, f'{file_name}-sentence_{token_num_per_sentence}')
    
    # for data_name in tqdm(os.listdir(data_dir)):
    #     img_concat(data_dir, output_dir, data_name, token_num_per_sentence)
    
    Parallel(n_jobs=64)(delayed(img_concat)(data_dir, output_dir, data_name, token_num_per_sentence)
                        for data_name in tqdm(os.listdir(data_dir)))
