"""
# Description:
#  Quality Assurance 
#
#  Copyright Heejun Shin
#  Artificial Intelligent Engineering Division
#  RadiSen Co. Ltd., Seoul, Korea
#  email : shj4901@radisentech.com

# running example 
python evaluate.py --mode classification --data drchoi --json_path /home/heejun/QA/choi_test_output.json

python evaluate.py --mode classification --data 3dr --json_path /home/heejun/QA/3dr_test_output.json

"""

from __future__ import annotations
import os
import math
import logging
from tqdm import tqdm
import numpy as np
import time
from typing import Dict, Tuple, Callable
import argparse

from scipy import io
import os
import torch.fft
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms as transforms
from scipy.io import savemat

from enum import Enum, auto
import argparse
from measure import *
from data import *
from save import *

import json

import warnings
warnings.filterwarnings('ignore')

#####################################################################################################
############################################ Parser #################################################
#####################################################################################################


def parse():
    parser = argparse.ArgumentParser(description='Evaluate')
    
    """ mode """
    parser.add_argument('--mode', type=str, default = 'classification',
                        choices=['classification', 'detection', 'segmentation'],
                        help = 'This argument is used in QA')
    
    """ dataset """
    parser.add_argument('--data',
                       default = [
                           'choi', 
                           'snu', 
                           'fit', 
                           'plco', 
                           'rsna', 
                           'stpeter'
                       ],
                       help = 'Which dataset will you use for evaluation')
    
    """ json path """
    parser.add_argument("--choi_out_json_path", "-j1", 
                        default = '/data/log/eval_log/classification/00362-eval-dsn/QA_test_output.json')
    parser.add_argument("--snu_out_json_path", "-j2", 
                        default = '/data/log/eval_log/classification/00360-eval-dsn/QA_test_output.json')
    parser.add_argument("--fit_out_json_path", "-j3", 
                        default = '/data/log/eval_log/classification/00356-eval-dsn/QA_test_output.json')
    parser.add_argument("--plco_out_json_path", "-j4", 
                        default = '/data/log/eval_log/classification/00358-eval-dsn/QA_test_output.json')
    parser.add_argument("--rsna_out_json_path", "-j5", 
                        default = '/data/log/eval_log/classification/00359-eval-dsn/QA_test_output.json')
    parser.add_argument("--stpeter_out_json_path", "-j6", 
                        default = '/data/log/eval_log/classification/00357-eval-dsn/QA_test_output.json')
    
    """ data label config """
    parser.add_argument("--classes",
        # default = [
        #     'atelectasis', 'lung opacity', 'effusion', 'nodule mass', 'hilar', 'fibrosis', 'cardiomegaly', 'pneumothorax', 'tuberculosis', 'normal', 'pneumonia', \
        #     'edema', 'nodulemasswocavitation', 'cavitarynodule', 'miliarynodule', 'fibrosisinfectionsequelae', 'fibrosisild', 'bronchiectasis', \
        #     'subcutaneousemphysema', 'pleuralthickening', 'pleuralcalcification', 'medical device',
        #           ],
                        
        default = [
            'atelectasis', 'lung opacity', 'effusion', 'nodule mass', 'hilar', 'fibrosis', 'pneumothorax', 'tuberculosis', 'pneumonia', 'edema','normal'
                  ],
        
        help="Provide Classes Here")
    
    """ evaluation config """
    parser.add_argument("--runs_dir","-dir",type = str, 
                        default = '/home/heejun/Projects/Quality-Assurance/projects/',
                        help = "where to save the results")
    
    parser.add_argument("--run_dir", type = str,
                        help = "if you want to specify project name")


    """ Miscellaneous """
    
    args = parser.parse_args()
    args.num_classes = len(args.classes)
    return args


def start(args):
    run_dir_path = set_run_dir_path(args, args.run_dir)
    # ----------------------------------- #
    #          QA parameter define        #
    # ----------------------------------- #
    print(separator())
    print("Quality Assurance : saving at {}".format(run_dir_path))
    print(separator())
    print("ArgPaser Info")
    for key, value in vars(args).items():
        print('{:15s}: {}'.format(key,value))
    print(separator())
    
    # ----------------------------------- #
    #               Data define           #
    # ----------------------------------- #
    print("Reading json files...")
    prediction, ground_truth = make_data(args)
    
    assert len(prediction) == len(ground_truth)
    

    
    # ----------------------------------- #
    #               Evaluation            #
    # ----------------------------------- #
    print("Evaluate...")
    if args.mode == 'classification':
        evaluator = eval_classification(args, ground_truth, prediction)
    
    elif args.mode == 'detection':
        pass
    
    # ----------------------------------- #
    #              Result save            #
    # ----------------------------------- #
    print("saving result...")
    if args.mode == 'classification':
        save_classification_results(args, evaluator, run_dir_path)
    
            

    
# ---------------------------------
# setting running directory
# ---------------------------------
def set_run_dir_path(args, run_dir = None):
    if run_dir == None:
        run_dir_name = "{:05d}-qa".format(next_run_id())
        run_dir = run_dir_name
        
    running_path = os.path.join(args.runs_dir, run_dir)
        
    if running_path != "" and not os.path.exists(running_path):
        os.makedirs(running_path)
        
    args.run_dir = run_dir
        
    return running_path

def next_run_id():
    run_ids = []
    if os.path.exists(args.runs_dir):
        for run_dir_path, _, _ in os.walk(args.runs_dir):
            run_dir = run_dir_path.split(os.sep)[-1]
            try:
                run_ids += [int(run_dir.split("-")[0])]
            except Exception:
                continue
    return max(run_ids) + 1 if len(run_ids) > 0 else 0

def save_output(inp, lab, output, fea, save_dir = str, detail_dir = str, file_name = str):
    out = {}
    out['inp'] = inp.cpu().detach().numpy()
    out['lab'] = lab.cpu().detach().numpy()

    for i in range(10):
        out['out{}'.format(i)] = output[i].cpu().detach().numpy()
        out['fea{}'.format(i)] = fea[i].cpu().detach().numpy()

    output_dir = os.path.join(save_dir, detail_dir)
    os.makedirs(output_dir, exist_ok=True)
    savemat(os.path.join(output_dir, file_name),out)
    
def test_data(args):
    dataset = InferWrapper(
        file_path = args.test_dataset,
        args = args,
        algorithm = args.algorithm,
        training_mode = False,
    )  

    _ = dataset[0]

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        num_workers= args.num_workers,
        pin_memory=True,
        shuffle=False,
    )

    return dataloader, dataset, len(dataset), dataset.data_info()
    
    

if __name__ == "__main__":
    args = parse()
    start(args)