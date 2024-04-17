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

import json
import csv

from config import *
from utils import *

from pycocotools.coco import COCO


global categories
categories = [
            {'id': 1, 'name': 'atelectasis', 'supercategory': 'atelectasis'},
            {'id': 2, 'name': 'lung opacity', 'supercategory': 'lung opacity'},
            {'id': 3, 'name': 'effusion', 'supercategory': 'effusion'},
            {'id': 4, 'name': 'nodule mass', 'supercategory': 'nodule mass'},
            {'id': 5, 'name': 'hilar', 'supercategory': 'hilar'},
            {'id': 6, 'name': 'fibrosis', 'supercategory': 'fibrosis'},
            {'id': 7, 'name': 'pneumothorax', 'supercategory': 'pneumothorax'},
            {'id': 8, 'name': 'cardiomegaly', 'supercategory': 'cardiomegaly'},
            {'id': 9, 'name': 'edema', 'supercategory': 'lung opacity'},
            {'id': 10, 'name': 'nodulemasswocavitation', 'supercategory': 'nodule mass'},
            {'id': 11, 'name': 'cavitarynodule', 'supercategory': 'nodule mass'},
            {'id': 12, 'name': 'miliarynodule', 'supercategory': 'nodule mass'},
            {'id': 13, 'name': 'fibrosisinfectionsequelae', 'supercategory': 'fibrosis'},
            {'id': 14, 'name': 'fibrosisild', 'supercategory': 'fibrosis'},
            {'id': 15, 'name': 'bronchiectasis', 'supercategory': 'bronchiectasis'},
            {'id': 16, 'name': 'emphysema', 'supercategory': 'emphysema'},
            {'id': 17, 'name': 'subcutaneousemphysema', 'supercategory': 'emphysema'},
            {'id': 18, 'name': 'pleuralthickening', 'supercategory': 'pleuralthickening'},
            {'id': 19, 'name': 'pleuralcalcification', 'supercategory': 'pleuralcalcification'},
            {'id': 20, 'name': 'medical device', 'supercategory': 'medical device'},
            {'id': 101, 'name': 'normal', 'supercategory': 'normal'},
            {'id': 102, 'name': 'pneumonia', 'supercategory': 'pneumonia'},
            {'id': 103, 'name': 'tuberculosis', 'supercategory': 'tuberculosis'},
            {'id': 104, 'name': 'others', 'supercategory': 'others'},
            {'id': -1, 'name': 'discard', 'supercategory': 'discard'}
        ]

def find_id_by_name(name):
    for category in categories:
        if category['name'] == name:
            return category['id']
    return None  # Return None if 'name' is not found



def make_data(args):
    ## Just paste from 'categories' in data COCO json file, if needs to be updated
    global template 
    template = args.classes
    template = [find_id_by_name(item) for item in template]
    # Example:   self.template = [1, 2, 3, 4, 5, 6, 8, 7, 103, 101]
    # For:   [atelectasis, lung_opacity, effusion, nodule_mass, hilar, fibrosis, cardiomegaly, pneumothorax, tb, normal]
    
    pred_dic = make_pred(args)
    label_dic = make_gt(args, pred_dic)
    
    # print(pred_dic.keys())
    # print(label_dic.keys())
    
    # print(label_dic)
    # print(pred_dic)
    
    return pred_dic, label_dic


def make_gt(args, pred_dic):
    if args.mode == 'classification':
        label_dict = {}
        for dataset in args.data:
            if dataset == 'snu':
                json_path = ['/home/heejun/Projects/Quality-Assurance/json/AWS_val.json', '/home/heejun/Projects/Quality-Assurance/json/LEE_unified.json']
                label_dict_generator = Label_dict_generator(args, json_path, list(pred_dic['snu'].keys()))
                _label_dic = label_dict_generator.get_label_dict()
                label_dict['snu'] = _label_dic
                
            if dataset == 'choi':
                json_path = ['/home/heejun/Projects/Quality-Assurance/json/indo_vnn_val.json', '/home/heejun/Projects/Quality-Assurance/json/indo_vnn_test.json']
                label_dict_generator = Label_dict_generator(args, json_path, list(pred_dic['choi'].keys()))
                _label_dic = label_dict_generator.get_label_dict()
                label_dict['choi'] = _label_dic
                
            if dataset == 'fit':
                json_path = ['/home/heejun/Projects/Quality-Assurance/json/FIT_unified.json']
                label_dict_generator = Label_dict_generator(args, json_path, list(pred_dic['fit'].keys()))
                _label_dic = label_dict_generator.get_label_dict()
                label_dict['fit'] = _label_dic
                
            if dataset == 'plco':
                json_path = ['/home/heejun/Projects/Quality-Assurance/json/PLCO_unified.json']
                label_dict_generator = Label_dict_generator(args, json_path, list(pred_dic['plco'].keys()))
                _label_dic = label_dict_generator.get_label_dict()
                label_dict['plco'] = _label_dic
                
            if dataset == 'rsna':
                json_path = ['/home/heejun/Projects/Quality-Assurance/json/RSNA_val.json']
                label_dict_generator = Label_dict_generator(args, json_path, list(pred_dic['rsna'].keys()))
                _label_dic = label_dict_generator.get_label_dict()
                label_dict['rsna'] = _label_dic
                
            if dataset == 'stpeter':
                json_path = ['/home/heejun/Projects/Quality-Assurance/json/StPeter_unified.json']
                label_dict_generator = Label_dict_generator(args, json_path, list(pred_dic['stpeter'].keys()))
                _label_dic = label_dict_generator.get_label_dict()
                label_dict['stpeter'] = _label_dic
                
            
                
    
    return label_dict


def make_pred(args):
    if args.mode == 'classification':
        pred_dic = {}
        for dataset in args.data:
            if dataset == 'snu':
                json_path = args.snu_out_json_path
                pred_dic['snu'] = make_pred_dic(json_path)
            if dataset == 'choi':
                json_path = args.choi_out_json_path
                pred_dic['choi'] = make_pred_dic(json_path)
            if dataset == 'fit':
                json_path = args.fit_out_json_path
                pred_dic['fit'] = make_pred_dic(json_path)
            if dataset == 'plco':
                json_path = args.plco_out_json_path
                pred_dic['plco'] = make_pred_dic(json_path)
            if dataset == 'rsna':
                json_path = args.rsna_out_json_path
                pred_dic['rsna'] = make_pred_dic(json_path)
            if dataset == 'stpeter':
                json_path = args.stpeter_out_json_path
                pred_dic['stpeter'] = make_pred_dic(json_path)
            
    return pred_dic


class Label_dict_generator():
    def __init__(self, args, json_path, pred_files):
        self.args = args
        self.json_path = json_path
        self.categories = categories
        self.pred_files = pred_files
        self.template = template
        
        
        self.num_classes = len(self.template)

        ### Going through dataset json files ###
        self.label_dict = {}
        for f_path in self.json_path: 
            # make label dict for each coco-json file
            self.make_label_dict(f_path)

        self.label_dict_lst = list(self.label_dict.keys())
        
    def get_label_template(self):
        # Creating a dictionary from the categories for quicker access
        categories_dict = {category['id']: category['name'] for category in self.categories}
        
        template_show = {}
        idx = 0
        # Iterating over the IDs and printing the corresponding names
        for id in self.template:
            if id in categories_dict:
                template_show[idx] = categories_dict[id]            
                idx += 1
        
        return template_show

    
    def list_files_with_extensions(self, directory, extensions, file_path_dict, idx):
        for root, _, files in os.walk(directory):
            for file in files:
                if any(file.endswith(ext) for ext in extensions):
                    if idx < self.max_data_num:
                        # files_lst.append(os.path.join(root, file))
                        _f = os.path.splitext(file)[0]
                        if _f in file_path_dict:
                            print("WARNING : {} is already present at {}".format(_f, file_path_dict[_f]))
                            # raise ## same filename error raise
                        file_path_dict[_f] = os.path.join(root, file)
                        idx += 1
        return file_path_dict, idx
    
    

    def filter_images_by_file_paths(self, image_ids, file_names):
        filtered_image_ids = []
        for img_id in image_ids:
            img_info = self.coco.imgs[img_id]
            # img_path = os.path.join(self.image_folder, img_info['file_name'])
            # img_name = img_info['file_name']
            img_name = os.path.splitext(img_info['file_name'])[0]
            if img_name in file_names:
                # if img_name == 'RVNNH_20210316_1549':
                #     print(img_id)
                filtered_image_ids.append(img_id)
        return filtered_image_ids

    def _get_valid_image_ids(self, image_ids):
        filtered_image_ids = []
        for img_id in image_ids:
            # print(img_id)
            
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)

            discard_annotations = [
                ann for ann in anns
                if (ann['category_id'] == 999 # 'discard' cat_id
                    or ann['category_id'] == -1) # old 'discard' cat_id
                    # or ann['category_id'] == 104) # 'others' cat_id
                and ann.get('score', 1) > 0 # if NOT 'score' negative (0) or missing_label (-1) 
            ]
            
            # if img_id == 61139:
            #     print(anns)
            #     print(discard_annotations)
                
            
            ## if there is "discard" category inside the annotation, the image is discarded
            ## if there is "other" category inside the annotation, it doesn't matter,
            ## but, if there is only "other" category in the image, it will be discarded at "make_label_dict" function
            ## if there is "other" category and the available labels, the image is included only with those labels (other category ignored).
            if not discard_annotations:
                filtered_image_ids.append(img_id)

        return filtered_image_ids

    
    
    def make_label_dict(self, f_path):
        ## Reading COCO json (COCO_Dataset initialize)
        self.coco = COCO(f_path)

        # self.category_id_to_name = {
        #     category['id']: category['name']
        #     for category in self.coco.dataset['categories']
        # }

        # All image IDs in COCO json file(s)
        self.image_ids = list(self.coco.imgs.keys()) 
        
        
        ## 학습시엔 필요하지만 여기서는 필요 없는 부분
#         # List of (coco) image IDs based on the filteration from file paths list
#         ## e.g. If an image is not available in the given dataset directory(-ies), its (coco) image ID will be omitted
#         self.image_ids = self.filter_images_by_file_paths(self.image_ids, self.pred_files)
#         # print('All matched files among .json and given directories: ', len(self.image_ids))

#         # # Discard the files with '-1' in 'category_id' for annotations the coco json file
#         self.image_ids = self._get_valid_image_ids(self.image_ids)
#         # print('Data length after discard in category_id: ', len(self.image_ids))

        for img_id in self.image_ids:

            img_info = self.coco.imgs[img_id]
            ''' e.g. (img_info=)
                {'id': 60001,          #img_id
                'width': 512,
                'height': 512,
                'file_name': 'RVNNH_20210316_0000.mat',
                'license': 1,
                'date_captured': '210316'}'''
            _name = img_info['file_name'] # image filename
            fname = os.path.splitext(_name)[0] # filename without extension
            _file = fname
            
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            annotations = self.coco.loadAnns(ann_ids)
            

            boxes = []
            labels = []
            masks = []
            ref = [0] * len(self.template)
            mask_labels = []
            loss_mask = [1] * len(self.template)

            for ann in annotations:
                if 'category_id' in ann and ann.get('score', 1) > 0:
                    # if a 'category_id' is available AND (its 'score' is positive OR 'score' not provided) 
                    # default scores -> -1: Not_evaluated, 0: Evaluated by doctor and found negative, 1-5: severity for positive case   
                    labels.append(ann['category_id'])
                    
                    ## bbox annotation 
                    if 'bbox' in ann and ann['bbox'] != []:
                        x, y, w, h = ann['bbox']
                    else:
                        x, y, w, h = 0,0,0,0
                    # Format [x_min, y_min, x_max, y_max]
                    # boxes.append([x, y, x + w, y + h])
                    boxes.append([x, y, w, h]) 
                    
                    ## segmetnation annotation
                    if 'segmentation' in ann and ann['segmentation'] != []:
                        masks.append(ann['segmentation'])
                    else:
                        masks.append([-1])

                if 'category_id' in ann and ann.get('score', 1) == -1:
                    # if a 'category_id' is available AND its 'score' is -1, intended for missing labels (Not_evaluated by Dr.)
                    mask_labels.append(ann['category_id'])



            #- 'ref' is multi-label class GT list. Example below
            #    [0, 1, 0, 0, 0, 1, 0, 0, 1, 0]
            for value in labels:
                if value in self.template:
                    index = self.template.index(value)
                    ref[index] = 1
            ref = np.array(ref)

            #- 'loss_mask' is multi-label mask. The model loss is computed for 1s, not for 0s. Example below for NODE21 data positive/negative sample (nodule annotation only)
            #    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
            for value in mask_labels:
                if value in self.template:
                    index = self.template.index(value)
                    loss_mask[index] = 0
            loss_mask = np.array(loss_mask)
            
            ## 학습 시에는 필요하지만 여기선 필요 없는 부분
            # Omit the data sample (image id) if all the findings are negative as well as 'normal' is negative
            # if 'normal' in self.args.classes and self.num_classes>1 and loss_mask[self.args.classes.index('normal')] == 1 and np.sum(ref) == 0:
            #     continue

            assert self.num_classes == len(ref)

            # Store information for each image in a dict -> dict_key=file_name without ext 
            # let's use only the reference
            # self.label_dict[fname] = {'path': _file}
            # self.label_dict[fname]['labels'] = labels
            # self.label_dict[fname]['boxes'] = boxes
            # self.label_dict[fname]['ref'] = ref
            # self.label_dict[fname]['loss_mask'] = loss_mask
            # self.label_dict[fname]['masks'] = masks
            self.label_dict[fname] = ref

    def get_label_dict(self):
        return self.label_dict
        
        


def count_data(args, dic):
    
    dic['count'] = {}
    
    for k, v in dic.items():
        if k == 'count':
            continue
        name = k # key is dr name
        
        positive = np.zeros((args.num_classes))
        negative = np.zeros((args.num_classes))
        
        dr_dic = v # value is dictionary of {filename : label}

        for _k, _v in dr_dic.items():
            p = [1 if __v == 1 else 0 for __v in _v]
            positive = positive + p

            n = [1 if __v == 0 else 0 for __v in _v]
            negative = negative + n
            
        dic['count'][name] = {'positive' : positive, 'negative' : negative}

    return dic

    
def make_pred_dic(pred_json):
    pred_dic = {}
    with open(pred_json, 'r') as p:
        p_data = json.load(p)
        for k, v in p_data.items():
            _filename = v['img']
            _output = v['out']
            _out = [float(_output[str(temp)]) for temp in template]
            _out = np.array(_out)

            pred_dic[_filename] = _out
            
    return pred_dic

            