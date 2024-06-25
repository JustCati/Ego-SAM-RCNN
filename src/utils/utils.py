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

