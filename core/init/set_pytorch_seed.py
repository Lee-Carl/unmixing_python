import os
import torch
import numpy as np
import random


def set_pytorch_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    else:
        print('WARNING: You have a CUDA device, so you should probably run with --cuda')
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
