import numpy as np
import re
import os
import time
import random
import cv2
from random import sample

from typing import Any, List, Dict
from torch import Tensor
from collections import OrderedDict

from utils import *

from tqdm import tqdm

import openpyxl
from PyPDF2 import PdfReader, PdfWriter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
from reportlab.lib.utils import ImageReader

from PIL import Image

from io import StringIO

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def save_classification_results(args, evaluator, save_dir):
    c = canvas.Canvas(
        filename = os.path.join(save_dir, '_.pdf'),
        pagesize = (600, 800)
    )

    ### writing canvas
    myFirstPage(args, c)
    
    mySecondPage(args, c)
    
    for _dataset, _evaluator in evaluator.items():
        if _dataset == 'choi' or _dataset == 'snu':
            WritePage(args, c, _dataset, _evaluator, save_dir)
        else:
            ## test datasets with one label
            WritePage2(args, c, _dataset, _evaluator, save_dir)
    
    c.save()
    
    ### radisen marked base page
    base_reader = PdfReader(open('base.pdf','rb'), strict=False)
    first_base_page = base_reader.pages[0]
    base_page = base_reader.pages[1]
    
    ### canvas pdf
    reader = PdfReader(open(os.path.join(save_dir, '_.pdf'), 'rb'))
    page_cnt = c.getPageNumber()
    
    writer = PdfWriter()
    
    for page_num in range(page_cnt -1):
        input_page = reader.pages[page_num]
        if page_num == 0:
            input_page.merge_page(first_base_page)
        else:
            input_page.merge_page(base_page)
        
        writer.add_page(input_page)
        
    with open(os.path.join(save_dir, 'result.pdf'),  'wb') as output:
        writer.write(output)
        
    os.remove(os.path.join(save_dir, '_.pdf'))
        
        
def myFirstPage(args, c):
    c.saveState() # 현재까지 canvas setting을 저장 --> restore 필요
    c.setFont('Helvetica', size = 15) # Helvetica / Times-Roman
    
    c.drawString(300, 490,"{}".format(args.run_dir))
    # c.restoreState()
    
    
    data = [['mode', args.mode, '', '', ''],
            ['project dir', args.runs_dir, '', '', ''],
            ['num_classes', args.num_classes, '', '', ''],
            # ['test_dataset', args.data, '', '', ''],
            # ['output path', args.json_path, '', '', ''],
           ]
    
    t = Table(data, colWidths = 90, rowHeights = 50)
    t.setStyle(TableStyle([
        # ('TEXTCOLOR',(0,0),(0,-1),colors.blue),
        ('ALIGN',(0,0),(-1,-1),'CENTER'),
        ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
        ('INNERGRID', (0,0), (-1,-1), 0.25, colors.black),
        ('BOX', (0,0), (-1,-1), 0.25, colors.black),
        ('BACKGROUND',(0,0),(0,-1), colors.lightgrey),
        ('SPAN',(1,0),(-1,0)),
        ('SPAN',(1,1),(-1,1)),
        ('SPAN',(1,2),(-1,2)),
        ('SPAN',(1,3),(-1,3)),
        ('SPAN',(1,4),(-1,4)),
        ('FONTSIZE', (1, 0), (1, -1), 15),
                            ]))
    t.wrapOn(c, 100, 100)
    t.drawOn(c, 100, 100)
    
    c.showPage()
    
def mySecondPage(args, c):
    c.saveState() # 현재까지 canvas setting을 저장 --> restore 필요
    c.setFont('Helvetica', size = 15) # Helvetica / Times-Roman
    
    c.drawString(70, 670, "{}".format("Abbreviation"))
    
    data = [
        ['Medical Terms', '', ''],
        ['atx', 'atelectasis', ''],
        ['lo', 'lung opacity', ''],
        ['eff', 'effusion', ''],
        ['nm', 'nodule/mass', ''],
        ['ha', 'hilar abnormality', ''],
        ['fib', 'fibrosis', ''],
        ['ptx', 'pneumothorax', ''],
        ['tb', 'tuberculosis', ''],
        ['pna', 'pneumonia', ''],
        ['ede', 'edema', ''],
        ['nor', 'normal', ''],
           ]
    
    t = Table(data, colWidths = 80, rowHeights = 40)
    t.setStyle(TableStyle([
        # ('TEXTCOLOR',(0,0),(0,-1),colors.blue),
        ('ALIGN',(0,0),(-1,-1),'CENTER'),
        ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
        ('INNERGRID', (0,0), (-1,-1), 0.25, colors.black),
        ('BOX', (0,0), (-1,-1), 0.25, colors.black),
        ('BACKGROUND',(0,0),(-1,0), colors.lightgrey),
        ## merging two column
        ('SPAN',(0,0),(-1,0)),
        ('SPAN',(1,1),(-1,1)),
        ('SPAN',(1,2),(-1,2)),
        ('SPAN',(1,3),(-1,3)),
        ('SPAN',(1,4),(-1,4)),
        ('SPAN',(1,5),(-1,5)),
        ('SPAN',(1,6),(-1,6)),
        ('SPAN',(1,7),(-1,7)),
        ('SPAN',(1,8),(-1,8)),
        ('SPAN',(1,9),(-1,9)),
        ('SPAN',(1,10),(-1,10)),
        ('SPAN',(1,11),(-1,11)),
        
        # ('FONTSIZE', (0, 0), (-1, 1), 15),
        # ('RIGHTPADDING', (2, 2), (-1, -1), 20),
                            ]))
    t.wrapOn(c, 60, 100)
    t.drawOn(c, 60, 100)
    
    data = [
        ['Performance Metric', '', ''],
        ['sen', 'sensitivity', ''],
        ['spe', 'specificity', ''],
        ['acc', 'accuracy', ''],
        ['ppv', 'positive prediction value', ''],
        ['npv', 'negative prediction value', ''],
        ['p-like', 'positive likelihood ratio', ''],
        ['n-like', 'negative likelihood ratio', ''],
        ['roc', 'receiver operating characteristic', ''],
        ['auc', 'area under roc curve', ''],
        ['', '', ''],
        ['', '', ''],
           ]
    
    t = Table(data, colWidths = 80, rowHeights = 40)
    t.setStyle(TableStyle([
        # ('TEXTCOLOR',(0,0),(0,-1),colors.blue),
        ('ALIGN',(0,0),(-1,-1),'CENTER'),
        ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
        ('INNERGRID', (0,0), (-1,-1), 0.25, colors.black),
        ('BOX', (0,0), (-1,-1), 0.25, colors.black),
        ('BACKGROUND',(0,0),(-1,0), colors.lightgrey),
        ('SPAN',(0,0),(-1,0)),
        ('SPAN',(1,1),(-1,1)),
        ('SPAN',(1,2),(-1,2)),
        ('SPAN',(1,3),(-1,3)),
        ('SPAN',(1,4),(-1,4)),
        ('SPAN',(1,5),(-1,5)),
        ('SPAN',(1,6),(-1,6)),
        ('SPAN',(1,7),(-1,7)),
        ('SPAN',(1,8),(-1,8)),
        ('SPAN',(1,9),(-1,9)),
        
        ('SPAN',(1,10),(-1,10)),
        ('SPAN',(1,11),(-1,11)),
        
        # ('FONTSIZE', (0, 0), (-1, 1), 15),
        # ('RIGHTPADDING', (2, 2), (-1, -1), 20)
                            ]))
    t.wrapOn(c, 300, 100)
    t.drawOn(c, 300, 100)
    
    c.showPage()
    
def WritePage(args, c, dataset, evaluator, save_dir):
    """
    맨위 글씨 높이 680
    좌측 끝 70
    표 좌측 끝 60

    세로 표나 제목 간 간격 40
    """
    ### caculate performance ###
    tpr, fpr = evaluator.ROC()
    threshs, sen, spe = evaluator.info()
    aucs = evaluator.eval()
    tp, fp, tn, fn = evaluator.tfpn()
    
    acc = (tp + tn) / (tp + tn + fp + fn)
    
    num_ex = evaluator.num_examples
    
    ### doctor config ###
    c.setFont('Helvetica', size = 15) # Helvetica / Times-Roman
    c.drawString(70, 680, "Dataset Name : {}".format(dataset))
    
    
    ### data table ###
    
    c.drawString(70, 650, "1. Dataset")
    class_names = ['class', 'atx', 'lo', 'eff', 'nm', 'ha', 'fib','ptx','tb', 'pna', 'ede','nor']
    data_table = [None for x in range(2)]
    data_table[0] = class_names
    data_table[1] = ['positive'] + list(num_ex.astype(np.uint16))
        
    t = Table(data_table, colWidths = 40, rowHeights = 20)
    t.setStyle(TableStyle([
        # ('TEXTCOLOR',(0,0),(0,-1),colors.blue),
        ('ALIGN',(0,0),(-1,-1),'CENTER'),
        ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
        ('INNERGRID', (0,0), (-1,-1), 0.25, colors.black),
        ('BOX', (0,0), (-1,-1), 0.25, colors.black),
        ('BACKGROUND',(0,0),(0,-1), colors.lightgrey),
        ('BACKGROUND',(0,0),(-1,0), colors.lightgrey),
        
    ]))
    
    t.wrapOn(c, 55, 550)
    t.drawOn(c, 55, 550)
    
    ########################## 
    ### ROC Curve : disease
    ###########################
    c.drawString(70, 510, "2-1. ROC Curve : findings ")
    line_color = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'olive']

    fig = plt.figure(figsize = (8, 8))
    ax0 = fig.add_subplot(1,1,1)

    for idx in range(args.num_classes - 4):
        t = tpr[idx]
        f = fpr[idx]
        ax0.plot(f, t, linestyle='solid', linewidth=1, color = line_color[idx], label = class_names[idx+1])
        # ax0.text(0.35, 0.75, "AUC = {:.3f}".format(aucs[idx]), fontsize = 15)

    ax0.set_title('ROC curve', fontsize = 20)
    ax0.plot([0.0,1.0], [0.0,1.0], linestyle='--',color = 'gray')
    ax0.plot([0.0,0.0], [0.0,1.0], linestyle='--', color = 'gray')
    ax0.plot([0.0,1.0], [1.0,1.0], linestyle='--', color = 'gray')
    ax0.set_xlabel('False Positive Rate', fontsize = 15)
    ax0.set_ylabel('True Positive Rate', fontsize = 15)
    ax0.legend(loc='lower right', fontsize = 15)

    fig.savefig(os.path.join(save_dir, 'roc{}.png'.format(dataset)))

    c.drawImage(os.path.join(save_dir, 'roc{}.png'.format(dataset)), x=60, y=30, width=470, height=470)

    os.remove(os.path.join(save_dir, 'roc{}.png'.format(dataset)))

    c.showPage() # new page
    
    ########################### 
    ### ROC Curve disease
    ###########################
    c.setFont('Helvetica', size = 15) # Helvetica / Times-Roman
    c.drawString(70, 650, "2-2. ROC Curve : disease")
    line_color = ['red', 'blue', 'green', 'yellow']

    fig = plt.figure(figsize = (8, 8))
    ax0 = fig.add_subplot(1,1,1)

    for idx in range(args.num_classes - 7):
        t = tpr[idx+7]
        f = fpr[idx+7]
        ax0.plot(f, t, linestyle='solid', linewidth=1, color = line_color[idx], label = class_names[idx+8])
        # ax0.text(0.35, 0.75, "AUC = {:.3f}".format(aucs[idx]), fontsize = 15)

    ax0.set_title('ROC curve', fontsize = 20)
    ax0.plot([0.0,1.0], [0.0,1.0], linestyle='--',color = 'gray')
    ax0.plot([0.0,0.0], [0.0,1.0], linestyle='--', color = 'gray')
    ax0.plot([0.0,1.0], [1.0,1.0], linestyle='--', color = 'gray')
    ax0.set_xlabel('False Positive Rate', fontsize = 15)
    ax0.set_ylabel('True Positive Rate', fontsize = 15)
    ax0.legend(loc='lower right', fontsize = 15)

    fig.savefig(os.path.join(save_dir, 'roc2{}.png'.format(dataset)))
    c.drawImage(os.path.join(save_dir, 'roc2{}.png'.format(dataset)), x=60, y=100, width=470, height=470)
    os.remove(os.path.join(save_dir, 'roc2{}.png'.format(dataset)))

    c.showPage() # new page
    
    ### doctor config ###
    c.setFont('Helvetica', size = 15) # Helvetica / Times-Roman
    # c.drawString(70, 680, "Doctor config : {}".format(doctor_name(dr_name)))
    
    ##############################
    ### performnace table : finding
    ###
    c.drawString(70, 680, "3-1. Performance : findings")

    col_names = ['class', 'cut-off', 'sen', 'spe', 'acc', 'ppv', 'npv', 'p-like', 'n-like', 'auc']
    per_table = []
    per_table.append(col_names)

    for idx in range(args.num_classes - 4):
        ## sen 0.9
        line = []

        i = find_nearest(sen[idx], 0.9)
        _thresh = round(threshs[i], 3)
        _tp = tp[idx][i]
        _fp = fp[idx][i]
        _tn = tn[idx][i]
        _fn = fn[idx][i]

        _sen = round(_tp / (_tp + _fn), 3)
        _spe = round(_tn / (_tn + _fp), 3)
        _ppv = round(_tp / (_tp + _fp), 3)
        _npv = round(_tn / (_tn + _fn), 3)
        _acc = round((_tp + _tn) / (_tp + _tn + _fp + _fn), 3)
        _plike = round(_sen / (1 - _spe), 3)
        _nlike = round((1-_sen) / _spe, 3)
        # _tfpn = [str(_tp), str(_fp), str(_tn), str(_fn)]

        _auc = round(aucs[idx], 3)

        line.append(class_names[idx+1])
        line.append(_thresh)
        # line.append('/'.join(_tfpn))
        line.append(_sen)
        line.append(_spe)
        line.append(_acc)
        line.append(_ppv)
        line.append(_npv)
        line.append(_plike)
        line.append(_nlike)
        line.append(_auc)

        per_table.append(line)


        ## spe 0.9
        line = []
        i = find_nearest(spe[idx], 0.9)
        _thresh = round(threshs[i], 3)
        _tp = tp[idx][i]
        _fp = fp[idx][i]
        _tn = tn[idx][i]
        _fn = fn[idx][i]

        _sen = round(_tp / (_tp + _fn), 3)
        _spe = round(_tn / (_tn + _fp), 3)
        _ppv = round(_tp / (_tp + _fp), 3)
        _npv = round(_tn / (_tn + _fn), 3)
        _acc = round((_tp + _tn) / (_tp + _tn + _fp + _fn), 3)
        _plike = round(_sen / (1 - _spe), 3)
        _nlike = round((1-_sen) / _spe, 3)
        # _tfpn = [str(_tp), str(_fp), str(_tn), str(_fn)]

        _auc = round(aucs[idx], 3)

        line.append(class_names[idx+1])
        line.append(_thresh)
        # line.append('/'.join(_tfpn))
        line.append(_sen)
        line.append(_spe)
        line.append(_acc)
        line.append(_ppv)
        line.append(_npv)
        line.append(_plike)
        line.append(_nlike)
        line.append(_auc)

        per_table.append(line)

        ## best accuracy
        line = []
        i = np.argmax(np.array(acc[idx]))
        _thresh = round(threshs[i], 3)
        _tp = tp[idx][i]
        _fp = fp[idx][i]
        _tn = tn[idx][i]
        _fn = fn[idx][i]

        _sen = round(_tp / (_tp + _fn), 3)
        _spe = round(_tn / (_tn + _fp), 3)
        _ppv = round(_tp / (_tp + _fp), 3)
        _npv = round(_tn / (_tn + _fn), 3)
        _acc = round((_tp + _tn) / (_tp + _tn + _fp + _fn), 3)
        _plike = round(_sen / (1 - _spe), 3)
        _nlike = round((1-_sen) / _spe, 3)
        # _tfpn = [str(_tp), str(_fp), str(_tn), str(_fn)]

        _auc = round(aucs[idx], 3)

        line.append(class_names[idx+1])
        line.append(_thresh)
        # line.append('/'.join(_tfpn))
        line.append(_sen)
        line.append(_spe)
        line.append(_acc)
        line.append(_ppv)
        line.append(_npv)
        line.append(_plike)
        line.append(_nlike)
        line.append(_auc)

        per_table.append(line)


    tt = Table(per_table, colWidths = 50, rowHeights = 20)
    tt.setStyle(TableStyle([
        # ('TEXTCOLOR',(0,0),(0,-1),colors.blue),
        ('ALIGN',(0,0),(-1,-1),'CENTER'),
        ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
        ('INNERGRID', (0,0), (-1,-1), 0.25, colors.black),
        ('BOX', (0,0), (-1,-1), 0.25, colors.black),
        ('BACKGROUND',(0,0),(0,-1), colors.lightgrey),
        ('BACKGROUND',(0,0),(-1,0), colors.lightgrey),
        ('SPAN',(0,1),(0,3)),
        ('SPAN',(0,4),(0,6)),
        ('SPAN',(0,7),(0,9)),
        ('SPAN',(0,10),(0,12)),
        ('SPAN',(0,13),(0,15)),
        ('SPAN',(0,16),(0,18)),
        ('SPAN',(0,19),(0,21)),
        ('SPAN',(0,22),(0,24)),
        ('SPAN',(0,25),(0,27)),
        ('SPAN',(0,28),(0,30)),
        ('SPAN',(-1,1),(-1,3)),
        ('SPAN',(-1,4),(-1,6)),
        ('SPAN',(-1,7),(-1,9)),
        ('SPAN',(-1,10),(-1,12)),
        ('SPAN',(-1,13),(-1,15)),
        ('SPAN',(-1,16),(-1,18)),
        ('SPAN',(-1,19),(-1,21)),
        ('SPAN',(-1,22),(-1,24)),
        ('SPAN',(-1,25),(-1,27)),
        ('SPAN',(-1,28),(-1,30)),
    ]))

    tt.wrapOn(c, 60, 150) # 가로 (좌측부터) / 세로 (아래부터)
    tt.drawOn(c, 60, 150)

    
    #############
    ## disease
    #############
    c.showPage()
    
    ### doctor config ###
    c.setFont('Helvetica', size = 15) # Helvetica / Times-Roman
    # c.drawString(70, 680, "Doctor config : {}".format(doctor_name(dr_name)))
    
    ### performance table ###
    c.drawString(70, 680, "3-2. Performance : disease")

    col_names = ['class', 'cut-off', 'sen', 'spe', 'acc', 'ppv', 'npv', 'p-like', 'n-like', 'auc']
    per_table = []
    per_table.append(col_names)

    for idx in range(args.num_classes - 7):
        ## sen 0.9
        line = []

        i = find_nearest(sen[idx+7], 0.9)
        _thresh = round(threshs[i], 3)
        _tp = tp[idx][i]
        _fp = fp[idx][i]
        _tn = tn[idx][i]
        _fn = fn[idx][i]

        _sen = round(_tp / (_tp + _fn), 3)
        _spe = round(_tn / (_tn + _fp), 3)
        _ppv = round(_tp / (_tp + _fp), 3)
        _npv = round(_tn / (_tn + _fn), 3)
        _acc = round((_tp + _tn) / (_tp + _tn + _fp + _fn), 3)
        _plike = round(_sen / (1 - _spe), 3)
        _nlike = round((1-_sen) / _spe, 3)
        # _tfpn = [str(_tp), str(_fp), str(_tn), str(_fn)]

        _auc = round(aucs[idx+7], 3)

        line.append(class_names[idx+8])
        line.append(_thresh)
        # line.append('/'.join(_tfpn))
        line.append(_sen)
        line.append(_spe)
        line.append(_acc)
        line.append(_ppv)
        line.append(_npv)
        line.append(_plike)
        line.append(_nlike)
        line.append(_auc)

        per_table.append(line)


        ## spe 0.9
        line = []
        i = find_nearest(spe[idx+7], 0.9)
        _thresh = round(threshs[i], 3)
        _tp = tp[idx][i]
        _fp = fp[idx][i]
        _tn = tn[idx][i]
        _fn = fn[idx][i]

        _sen = round(_tp / (_tp + _fn), 3)
        _spe = round(_tn / (_tn + _fp), 3)
        _ppv = round(_tp / (_tp + _fp), 3)
        _npv = round(_tn / (_tn + _fn), 3)
        _acc = round((_tp + _tn) / (_tp + _tn + _fp + _fn), 3)
        _plike = round(_sen / (1 - _spe), 3)
        _nlike = round((1-_sen) / _spe, 3)
        # _tfpn = [str(_tp), str(_fp), str(_tn), str(_fn)]

        _auc = round(aucs[idx+7], 3)

        line.append(class_names[idx+8])
        line.append(_thresh)
        # line.append('/'.join(_tfpn))
        line.append(_sen)
        line.append(_spe)
        line.append(_acc)
        line.append(_ppv)
        line.append(_npv)
        line.append(_plike)
        line.append(_nlike)
        line.append(_auc)

        per_table.append(line)

        ## best accuracy
        line = []
        i = np.argmax(np.array(acc[idx+7]))
        _thresh = round(threshs[i], 3)
        _tp = tp[idx][i]
        _fp = fp[idx][i]
        _tn = tn[idx][i]
        _fn = fn[idx][i]

        _sen = round(_tp / (_tp + _fn), 3)
        _spe = round(_tn / (_tn + _fp), 3)
        _ppv = round(_tp / (_tp + _fp), 3)
        _npv = round(_tn / (_tn + _fn), 3)
        _acc = round((_tp + _tn) / (_tp + _tn + _fp + _fn), 3)
        _plike = round(_sen / (1 - _spe), 3)
        _nlike = round((1-_sen) / _spe, 3)
        # _tfpn = [str(_tp), str(_fp), str(_tn), str(_fn)]

        _auc = round(aucs[idx+7], 3)

        line.append(class_names[idx+8])
        line.append(_thresh)
        # line.append('/'.join(_tfpn))
        line.append(_sen)
        line.append(_spe)
        line.append(_acc)
        line.append(_ppv)
        line.append(_npv)
        line.append(_plike)
        line.append(_nlike)
        line.append(_auc)

        per_table.append(line)


    tt = Table(per_table, colWidths = 50, rowHeights = 20)
    tt.setStyle(TableStyle([
        # ('TEXTCOLOR',(0,0),(0,-1),colors.blue),
        ('ALIGN',(0,0),(-1,-1),'CENTER'),
        ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
        ('INNERGRID', (0,0), (-1,-1), 0.25, colors.black),
        ('BOX', (0,0), (-1,-1), 0.25, colors.black),
        ('BACKGROUND',(0,0),(0,-1), colors.lightgrey),
        ('BACKGROUND',(0,0),(-1,0), colors.lightgrey),
        ('SPAN',(0,1),(0,3)),
        ('SPAN',(0,4),(0,6)),
        ('SPAN',(0,7),(0,9)),
        ('SPAN',(0,10),(0,12)),
        ('SPAN',(0,13),(0,15)),
        ('SPAN',(-1,1),(-1,3)),
        ('SPAN',(-1,4),(-1,6)),
        ('SPAN',(-1,7),(-1,9)),
        ('SPAN',(-1,10),(-1,12))
    ]))

    tt.wrapOn(c, 60, 350)
    tt.drawOn(c, 60, 350)

    c.showPage()
        
        
     
    
def WritePage2(args, c, dataset, evaluator, save_dir):
    """
    맨위 글씨 높이 680
    좌측 끝 70
    표 좌측 끝 60

    세로 표나 제목 간 간격 40
    """
    ### caculate performance ###
    tpr, fpr = evaluator.ROC()
    threshs, sen, spe = evaluator.info()
    aucs = evaluator.eval()
    tp, fp, tn, fn = evaluator.tfpn()
    
    acc = (tp + tn) / (tp + tn + fp + fn)
    
    num_ex = evaluator.num_examples
    total_num = tp[0][0] + tn[0][0] + fp[0][0] + fn[0][0]
    
    if dataset == 'fit':
        target_class = ('tb', 7)
    elif dataset == 'plco':
        target_class = ('nm', 3)
    elif dataset == 'rsna':
        target_class = ('pna', 8)
    elif dataset == 'stpeter':
        target_class = ('tb', 7)
        
    
    ### doctor config ###
    c.setFont('Helvetica', size = 15) # Helvetica / Times-Roman
    c.drawString(70, 680, "Dataset Name : {}".format(dataset))
    
    ### data table ###
    c.drawString(70, 650, "1. Dataset")
    class_names = ['class', target_class[0], 'total']
    data_table = [None for x in range(2)]
    data_table[0] = class_names
    data_table[1] = ['positive'] + [list(num_ex.astype(np.uint16))[target_class[1]]] + [int(total_num)]
    
    t = Table(data_table, colWidths = 40, rowHeights = 20)
    t.setStyle(TableStyle([
        # ('TEXTCOLOR',(0,0),(0,-1),colors.blue),
        ('ALIGN',(0,0),(-1,-1),'CENTER'),
        ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
        ('INNERGRID', (0,0), (-1,-1), 0.25, colors.black),
        ('BOX', (0,0), (-1,-1), 0.25, colors.black),
        ('BACKGROUND',(0,0),(0,-1), colors.lightgrey),
        ('BACKGROUND',(0,0),(-1,0), colors.lightgrey),
        
    ]))
    
    t.wrapOn(c, 55, 590)
    t.drawOn(c, 55, 590)
    
    ##############################
    ### performnace table : finding
    ##############################
    c.drawString(70, 560, "2. Performance")

    col_names = ['class', 'cut-off', 'sen', 'spe', 'acc', 'ppv', 'npv', 'p-like', 'n-like', 'auc']
    per_table = []
    per_table.append(col_names)

    ## sen 0.9
    line = []

    i = find_nearest(sen[target_class[1]], 0.9)
    _thresh = round(threshs[i], 3)
    _tp = tp[target_class[1]][i]
    _fp = fp[target_class[1]][i]
    _tn = tn[target_class[1]][i]
    _fn = fn[target_class[1]][i]

    _sen = round(_tp / (_tp + _fn), 3)
    _spe = round(_tn / (_tn + _fp), 3)
    _ppv = round(_tp / (_tp + _fp), 3)
    _npv = round(_tn / (_tn + _fn), 3)
    _acc = round((_tp + _tn) / (_tp + _tn + _fp + _fn), 3)
    _plike = round(_sen / (1 - _spe), 3)
    _nlike = round((1-_sen) / _spe, 3)
    # _tfpn = [str(_tp), str(_fp), str(_tn), str(_fn)]

    _auc = round(aucs[target_class[1]], 3)

    line.append(class_names[1])
    line.append(_thresh)
    # line.append('/'.join(_tfpn))
    line.append(_sen)
    line.append(_spe)
    line.append(_acc)
    line.append(_ppv)
    line.append(_npv)
    line.append(_plike)
    line.append(_nlike)
    line.append(_auc)

    per_table.append(line)


    ## spe 0.9
    line = []
    i = find_nearest(spe[target_class[1]], 0.9)
    _thresh = round(threshs[i], 3)
    _tp = tp[target_class[1]][i]
    _fp = fp[target_class[1]][i]
    _tn = tn[target_class[1]][i]
    _fn = fn[target_class[1]][i]

    _sen = round(_tp / (_tp + _fn), 3)
    _spe = round(_tn / (_tn + _fp), 3)
    _ppv = round(_tp / (_tp + _fp), 3)
    _npv = round(_tn / (_tn + _fn), 3)
    _acc = round((_tp + _tn) / (_tp + _tn + _fp + _fn), 3)
    _plike = round(_sen / (1 - _spe), 3)
    _nlike = round((1-_sen) / _spe, 3)
    # _tfpn = [str(_tp), str(_fp), str(_tn), str(_fn)]

    _auc = round(aucs[target_class[1]], 3)

    line.append(class_names[1])
    line.append(_thresh)
    # line.append('/'.join(_tfpn))
    line.append(_sen)
    line.append(_spe)
    line.append(_acc)
    line.append(_ppv)
    line.append(_npv)
    line.append(_plike)
    line.append(_nlike)
    line.append(_auc)

    per_table.append(line)

    ## best accuracy
    line = []
    i = np.argmax(np.array(acc[target_class[1]]))
    _thresh = round(threshs[i], 3)
    _tp = tp[target_class[1]][i]
    _fp = fp[target_class[1]][i]
    _tn = tn[target_class[1]][i]
    _fn = fn[target_class[1]][i]

    _sen = round(_tp / (_tp + _fn), 3)
    _spe = round(_tn / (_tn + _fp), 3)
    _ppv = round(_tp / (_tp + _fp), 3)
    _npv = round(_tn / (_tn + _fn), 3)
    _acc = round((_tp + _tn) / (_tp + _tn + _fp + _fn), 3)
    _plike = round(_sen / (1 - _spe), 3)
    _nlike = round((1-_sen) / _spe, 3)
    # _tfpn = [str(_tp), str(_fp), str(_tn), str(_fn)]

    _auc = round(aucs[target_class[1]], 3)

    line.append(class_names[1])
    line.append(_thresh)
    # line.append('/'.join(_tfpn))
    line.append(_sen)
    line.append(_spe)
    line.append(_acc)
    line.append(_ppv)
    line.append(_npv)
    line.append(_plike)
    line.append(_nlike)
    line.append(_auc)

    per_table.append(line)

    tt = Table(per_table, colWidths = 50, rowHeights = 20)
    tt.setStyle(TableStyle([
        # ('TEXTCOLOR',(0,0),(0,-1),colors.blue),
        ('ALIGN',(0,0),(-1,-1),'CENTER'),
        ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
        ('INNERGRID', (0,0), (-1,-1), 0.25, colors.black),
        ('BOX', (0,0), (-1,-1), 0.25, colors.black),
        ('BACKGROUND',(0,0),(0,-1), colors.lightgrey),
        ('BACKGROUND',(0,0),(-1,0), colors.lightgrey),
        ('SPAN',(0,1),(0,3)),
        ('SPAN',(-1,1),(-1,3)),
    ]))

    tt.wrapOn(c, 55, 450) # 가로 (좌측부터) / 세로 (아래부터)
    tt.drawOn(c, 55, 450)
    
    ########################## 
    ### ROC Curve : disease
    ###########################
    c.drawString(70, 410, "3. ROC Curve ")
    line_color = ['blue']

    fig = plt.figure(figsize = (8, 8))
    ax0 = fig.add_subplot(1,1,1)

    t = tpr[target_class[1]]
    f = fpr[target_class[1]]
    ax0.plot(f, t, linestyle='solid', linewidth=1, color = line_color[0], label = class_names[1])
    # ax0.text(0.35, 0.75, "AUC = {:.3f}".format(aucs[idx]), fontsize = 15)

    ax0.set_title('ROC curve', fontsize = 20)
    ax0.plot([0.0,1.0], [0.0,1.0], linestyle='--',color = 'gray')
    ax0.plot([0.0,0.0], [0.0,1.0], linestyle='--', color = 'gray')
    ax0.plot([0.0,1.0], [1.0,1.0], linestyle='--', color = 'gray')
    ax0.set_xlabel('False Positive Rate', fontsize = 15)
    ax0.set_ylabel('True Positive Rate', fontsize = 15)
    ax0.legend(loc='lower right', fontsize = 15)

    fig.savefig(os.path.join(save_dir, 'roc{}.png'.format(dataset)))

    c.drawImage(os.path.join(save_dir, 'roc{}.png'.format(dataset)), x=130, y=30, width=350, height=350)

    os.remove(os.path.join(save_dir, 'roc{}.png'.format(dataset)))

    c.showPage() # new page
    
        
    
    
    
    