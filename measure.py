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

from utils import *

from tqdm import tqdm

def eval_classification(args, gt, pred):
    print("testing {} different dataset".format(len(gt.keys())))
    for k in gt.keys():
        print("- ", k)
    print("{}-class classification".format(len(args.classes)))
    
    eval_dic = {}
    
    print(separator())
    
    for dataset, label_dic in pred.items():
        print('{} dataset'.format(dataset))
#         positive_num = count_dic[dr_name]['positive']
#         negative_num = count_dic[dr_name]['negative']
        
#         for i in range(args.num_classes):
#             if i == args.num_classes - 1 :
#                 print("{}/{}  ".format(int(positive_num[i]), int(negative_num[i])))
#             else:
#                 print("{}/{}  ".format(int(positive_num[i]), int(negative_num[i])), end=' ')
            
        evaluator = multi_auc_calculator(len(args.classes))
        
        for k, v in tqdm(label_dic.items()):
            file_name = k
            prediction = v
            
            label = gt[dataset][file_name]
            evaluator.add(prediction, label)
            
        
        eval_dic[dataset] = evaluator

        
    return eval_dic
        
        
        
    
class multi_auc_calculator():
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.threshs = np.arange(0.,1.006,0.005)
        self.sensitivities = np.zeros((self.num_classes, int(self.threshs.shape[0])))
        self.specificities = np.zeros((self.num_classes, int(self.threshs.shape[0])))
        self.FNs = np.zeros((self.num_classes, int(self.threshs.shape[0])))
        self.FPs = np.zeros((self.num_classes, int(self.threshs.shape[0])))
        self.TPs = np.zeros((self.num_classes, int(self.threshs.shape[0])))
        self.TNs = np.zeros((self.num_classes, int(self.threshs.shape[0])))  
        self.TPR = np.zeros((self.num_classes, int(self.threshs.shape[0])))
        self.FPR = np.zeros((self.num_classes, int(self.threshs.shape[0])))
        self.sen_threshes = np.zeros((self.num_classes))
        self.spe_threshes = np.zeros((self.num_classes))
        self.num_examples = np.zeros((self.num_classes))
    
    def add(self, pred, lab):
        self.num_examples += lab
        
        pred = torch.Tensor(pred)
        lab = torch.Tensor(lab)

        preds = torch.zeros((len(self.threshs), self.num_classes))
        
        for i in range(len(self.threshs)):
            for j in range(self.num_classes):
                if pred[j] >= self.threshs[i]: pred_ = torch.tensor([1])
                else: pred_ = torch.tensor([0])

                preds[i][j] = pred_
                

        target_mat = lab * torch.ones_like(preds)
        
        correct_mat = preds.eq(target_mat)

        ### Saving the results for each cases (FN, TN, FP, TP) ###
        for i in range(self.num_classes):
            if lab[i] == -1:
                continue

            elif lab[i] == 1:
                self.TPs[i] += correct_mat[:, i].numpy()
                self.FNs[i] += 1 - correct_mat[:, i].numpy()
            else:
                self.TNs[i] += correct_mat[:, i].numpy()
                self.FPs[i] += 1-correct_mat[:, i].numpy()
                
    def get_sen_spe(self):
        for i in range(self.num_classes):
            if np.all(self.TPs[i] + self.FNs[i]) ==0:
                self.sensitivities[i] = 0
                self.specificities[i] = 0
            else:
                self.sensitivities[i] = (self.TPs[i]) / (self.TPs[i] + self.FNs[i])
                self.specificities[i] = (self.TNs[i]) / (self.TNs[i] + self.FPs[i])
            
        return self.sensitivities, self.specificities
    
    def tfpn(self):
        return (self.TPs, self.FPs, self.TNs, self.FNs)

            
        
    def info(self):
        sensitivities, specificities = self.get_sen_spe()
        return (self.threshs, sensitivities, specificities)
    
    def ROC(self):
        for i in range(self.num_classes):
            if np.all(self.TPs[i] + self.FNs[i]) ==0:
                self.TPR[i] = 0
                self.FPR[i] = 0
            else:
                
                self.TPR[i] = self.TPs[i] / (self.TPs[i] + self.FNs[i])
                self.FPR[i] = self.FPs[i] / (self.TNs[i] + self.FPs[i])

        return (self.TPR, self.FPR)
        
    
    def eval(self):
        aucs = np.zeros((self.num_classes))
        sensitivities, specificities = self.get_sen_spe()
        for i in range(self.num_classes):
            avg_sensitivity = (sensitivities[i][:-1]+sensitivities[i][1:])/2
            diff_specificity = np.abs(specificities[i][:-1]-specificities[i][1:])
            auc = np.sum(avg_sensitivity*diff_specificity)
            aucs[i] = auc
        
        return aucs

    
def calculate_acc(pred: Tensor, label: Tensor):
    with torch.no_grad():
        __, Pred_label = torch.max(pred, 1)
    acc = (Pred_label == Real_label)
    return acc

def multi_calculate_acc(pred: Tensor, label: Tensor, num_classes):
    ## pcam output은 class개의 element가 있는 list, 
    ## original eff는 (batch, class)의 tensor
    acc_multi = torch.zeros((num_classes, label.shape[0]))
    with torch.no_grad():
        for i in range(num_classes):
            pred_label = torch.sigmoid(pred[i].view(-1)).ge(0.5).float()
            acc = (pred_label == label[:,i])
            acc_multi[i] = acc

    return acc_multi



def postprocess(logits, logit_maps, class_threshold = 0.1, heatmap_threshold = 0.1, iou_threshold = 0.5):
    """
    class_threshold : only make heatmap when class score is over
    heatmap_threshold : for clustering heatmap, delete lower pixels
    iou_threshold : 
    
    """
    resize_512 = transforms.Resize(size=(512, 512))
    
    out_boxes = []
    for i in range(10):
        if i ==8 or i == 9: continue
        
        score = torch.sigmoid(logits[i]).cpu().detach()
        outmap = logit_maps[i].cpu().detach()
            
        if score > class_threshold:
            ## making heatmap
            outmap = torch.sigmoid(outmap)
            outmap = resize_512(outmap.unsqueeze(0)).squeeze().numpy()
            
            ## making mask
            mask = (outmap > heatmap_threshold)
            mask = mask.astype('uint8')
            mask = cv2.dilate(mask, np.ones((7, 7), np.uint8), iterations=1)  # 애매하게 떨어져 있는 cluster 붙여주기 위해 dilation 한번
            mask = cv2.erode(mask, np.ones((7, 7), np.uint8), iterations=1)  # size 기준으로 작은 조각들 없애므로 erode 많이 할 필요 없음
            components, output, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=4)
            
            
            ## 각각의 cluster에 대해
            for k in range(1, components):
                box = stats[k]
                
                x1 = box[0]
                y1 = box[1]
                x2 = x1 + box[2]
                y2 = y1 + box[3]
                area = box[4]
                
                sub_map = np.where(output == k, 1, 0)
                
                masked_map = outmap * sub_map
                map_score = masked_map.sum() / area
                
                
                # if sub_map.sum() > 2000: ## 너무 작은 cluster는 무시
                #     sub_maps.append((sub_map, [i+1])) ## cluster와 finding label을 묶어서 저장
                
                out_boxes.append({"roi" : np.array((x1,y1,x2,y2)),
                                  "class_id" : np.array((i)),
                                  "score" : np.array((map_score))})
    
    return out_boxes
            

        
