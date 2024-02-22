import json
import random

import multiprocess
import numpy as np
import mindspore
from mindspore import nn

from .load import load_model


def init_seed():
    # set seed
    import mindspore
    random_seed = 1
    random.seed(42)
    mindspore.set_grad_enabled(False)
    mindspore.manual_seed(random_seed)
    mindspore.cuda.manual_seed(random_seed)
    mindspore.backends.cudnn.deterministic = True
    mindspore.backends.cudnn.benchmark = False
    np.random.seed(random_seed)


class ParallelWrapper(nn.Module):
    def __init__(self, vq_model, func='encode'):
        super().__init__()
        self.vq_model = vq_model
        self.func = func
    
    def forward(self, x):
        return getattr(self.vq_model, self.func)(x)


def init_vqgan_encoder(model_name_or_path, device):
    init_seed()
    vq_model = load_model(model_name_or_path)
    vq_model = vq_model.to(device).eval()
    
    print('vq_model device:', vq_model.device)
    
    encoder = ParallelWrapper(vq_model)
    
    return encoder


def get_multiprocess():
    multiprocess.set_start_method('spawn', force=True)
    mindspore.utils.data.dataloader.python_multiprocessing = multiprocess
    new_multiprocess_ctx = multiprocess.get_context()
    return new_multiprocess_ctx


def dumps(data):
    seqlen = len(data)
    saved_bin = str.encode(json.dumps(dict(tokens=data)) + "\n")
    return {"bin": saved_bin, "length": seqlen}
