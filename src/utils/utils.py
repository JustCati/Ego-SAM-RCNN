import os
import random
import numpy as np

import torch
from torch.utils import data
import torch.backends.cudnn as cudnn



def getDevice():
    device = None
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    return device


def fix_random_seed(seed):
    rng_generator = torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    cudnn.enabled = True
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    #! Commented because roi__align has a deterministic versione very memory hungry 
    # torch.use_deterministic_algorithms(True) #! https://github.com/pytorch/vision/issues/8168#issuecomment-1890599205
    return rng_generator


