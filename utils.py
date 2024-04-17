import numpy as np
import re
import os
import time
import random
import cv2
from random import sample

import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torch.utils.data import DataLoader, SubsetRandomSampler

from typing import Any, List, Dict
from torch import Tensor
from collections import OrderedDict

def list_constants(clazz: Any, private: bool = False) -> List[Any]:
    variables = [i for i in dir(clazz) if not callable(i)]
    regex = re.compile(r"^{}[A-Z0-9_]*$".format("" if private else "[A-Z]"))
    names = list(filter(regex.match, variables))
    values = [clazz.__dict__[name] for name in names]
    return values


def seconds_to_dhms(seconds: float) -> str:
    s = seconds % 60
    m = (seconds // 60) % 60
    h = seconds // (60 * 60) % 24
    d = seconds // (60 * 60 * 24)
    times = [(d, "d "), (h, "h "), (m, "m "), (s, "s")]
    time_str = ""
    for t, char in times:
        time_str += "{:02}{}".format(int(t), char)
    return time_str


class Metric:
    def __init__(self, batched: bool = True, collapse: bool = True):
        self.reset()
        self.batched = batched
        self.collapse = collapse

    def add(self, value: Tensor):
        n = value.shape[0] if self.batched else 1
        # n = value.shape[0] if len(value.shape) == 2 else 1
        
        if self.collapse:
            data_start = 1 if self.batched else 0
            mean_dims = list(range(data_start, len(value.shape)))
            if len(mean_dims) > 0:
                value = torch.mean(value, dim=mean_dims)
        if self.batched:
            value = torch.sum(value, dim=0)
        if self.total is None:
            self.total = value
        else:
            self.total += value
            
        self.n += n
        

    def __add__(self, value: Tensor):
        self.add(value)
        return self

    def accumulated(self, reset: bool = False):
        if self.n == 0:
            return None
        acc = self.total / self.n
        if reset:
            self.reset()
        return acc

    def reset(self):
        self.total = None
        self.n = 0

    def empty(self) -> bool:
        return self.n == 0


class MetricDict(OrderedDict):
    def __missing__(self, key):
        self[key] = value = Metric()
        return value


def separator(cols=80) -> str:
    return "#" * cols


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 'True', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'False', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        
        
def find_nearest(array, value):
    array = np.asarray(array)
    if len(array.shape) == 1:
        idx = (np.abs(array-value)).argmin()
        return idx
    
    elif len(array.shape) == 2:
        idx = (np.abs(array-value)).argmin()
        i = idx // array.shape[0]
        j = idx % array.shape[1]
        return i, j
    else:
        print("array shape error")
        