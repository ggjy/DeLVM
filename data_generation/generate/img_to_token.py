import json
import os
from itertools import chain
from os.path import join

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from vqgan.utils import init_vqgan_encoder


def data_loader_to_token(encoder, data_loader, device):
    cu = 0
    data_bin_list = []
    cu_seq_len_list = []
    for _data in tqdm(data_loader):
        _data = _data.to(device)
        
        if _data.dim() == 5:
            data_list = list(torch.split(_data, 1, dim=0))
            data_list = [i.squeeze(dim=0) for i in data_list]
            data = torch.cat(data_list, dim=0)
        else:
            data = _data
        
        _, out_tokens = encoder(data)
        
        indices_list = list(torch.split(out_tokens, 2, dim=0))
        
        for indices in indices_list:
            tokens = list(chain(*indices.tolist()))
            seq_len = len(tokens)
            saved_bin = str.encode(json.dumps(dict(tokens=tokens)) + "\n")
            
            data_bin_list.append(saved_bin)
            cu_seq_len_list.append((cu, seq_len))
            cu += len(saved_bin)
    return data_bin_list, cu_seq_len_list


def save_bin_and_meta_file(out_dir, data_bin_list, cu_seq_len_list):
    os.makedirs(out_dir, exist_ok=True)
    out_bin = join(out_dir, "train.bin")
    out_meta = join(out_dir, "train.bin.meta")
    
    with open(out_bin, "wb+") as bin_file:
        bin_file.writelines(data_bin_list)
    
    cu_seq_len_list = np.array(cu_seq_len_list, dtype=np.int64)
    with open(out_meta, "wb+") as meta_file:
        np.save(meta_file, cu_seq_len_list)


def img_to_token(args, data_loader, out_dir, device=None):
    encoder = init_vqgan_encoder(args.model_name_or_path, device)
    
    if args.dp_mode:
        encoder = nn.DataParallel(encoder)
    
    data_bin_list, cu_seq_len_list = data_loader_to_token(encoder, data_loader, device)
    save_bin_and_meta_file(out_dir, data_bin_list, cu_seq_len_list)
