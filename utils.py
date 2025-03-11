import math
import os
import pickle
import re
import warnings
warnings.filterwarnings('ignore')
import lifelines
from lifelines.utils import concordance_index
from lifelines import CoxPHFitter
from lifelines.datasets import load_regression_dataset
from lifelines.utils import k_fold_cross_validation
from lifelines.statistics import logrank_test
from imblearn.over_sampling import RandomOverSampler
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import numpy as np
import pandas as pd
from PIL import Image
import pylab
import scipy
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import average_precision_score, auc, f1_score, roc_curve, roc_auc_score
from sklearn.preprocessing import LabelBinarizer
from scipy import interp
mpl.rcParams['axes.linewidth'] = 3 #set the value globally
import torch
import torch.nn as nn
from torch.nn import init, Parameter
from torch.utils.data._utils.collate import *
from torch.utils.data.dataloader import default_collate
import torch_geometric
from torchvision import datasets, transforms
import skimage.feature
from networks import *
import cv2
import networkx as nx
from pathlib import Path
import random
import io
from absl import flags
from torch.optim.optimizer import Optimizer
def init_weights(net, init_type='orthogonal', init_gain=0.02):
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_max_weights(module):
    for m in module.modules():
        if type(m) == nn.Linear:
            stdv = 1. / math.sqrt(m.weight.size(1))
            m.weight.data.normal_(0, stdv)
            m.bias.data.zero_()
def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[1,0]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)           # multi-GPUs

    if init_type != 'max' and init_type != 'none':
        print("Init Type:", init_type)
        init_weights(net, init_type, init_gain=init_gain)
    elif init_type == 'none':
        print("Init Type: Not initializing networks.")
    elif init_type == 'max':
        print("Init Type: Self-Normalizing Weights")
    return net

def unfreeze_unimodal(opt, model, epoch):
    if opt.mode == 'graphomic':
        if epoch == 5:
            dfs_unfreeze(model.module.omic_net)
            print("Unfreezing Omic")
        if epoch == 5:
            dfs_unfreeze(model.module.grph_net)
            print("Unfreezing Graph")
    elif opt.mode == 'pathomic':
        if epoch == 5:
            dfs_unfreeze(model.module.omic_net)
            print("Unfreezing Omic")
    elif opt.mode == 'pathgraph':
        if epoch == 5:
            dfs_unfreeze(model.module.grph_net)
            print("Unfreezing Graph")
    elif opt.mode == "pathgraphomic":
        if epoch == 5:
            dfs_unfreeze(model.module.omic_net)
            print("Unfreezing Omic")
        if epoch == 5:
            dfs_unfreeze(model.module.grph_net)
            print("Unfreezing Graph")
    elif opt.mode == "omicomic":
        if epoch == 5:
            dfs_unfreeze(model.module.omic_net)
            print("Unfreezing Omic")
    elif opt.mode == "graphgraph":
        if epoch == 5:
            dfs_unfreeze(model.module.grph_net)
            print("Unfreezing Graph")


def dfs_freeze(model):
    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = False
        dfs_freeze(child)


def dfs_unfreeze(model):
    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = True
        dfs_unfreeze(child)


def print_if_frozen(module):
    for idx, child in enumerate(module.children()):
        for param in child.parameters():
            if param.requires_grad == True:
                print("Learnable!!! %d:" % idx, child)
            else:
                print("Still Frozen %d:" % idx, child)


def unfreeze_vgg_features(model, epoch):
    epoch_schedule = {30:45}
    unfreeze_index = epoch_schedule[epoch]
    for idx, child in enumerate(model.features.children()):
        if idx > unfreeze_index:
            print("Unfreezing %d:" %idx, child)
            for param in child.parameters(): 
                param.requires_grad = True
        else:
            print("Still Frozen %d:" %idx, child)
            continue

def mixed_collate(batch):
    elem = batch[0]
    elem_type = type(elem)    
    transposed = zip(*batch)
    return [Batch.from_data_list(samples, []) if type(samples[0]) is torch_geometric.data.data.Data else default_collate(samples) for samples in transposed]

def CoxLoss(survtime, censor, hazard_pred, device):
    current_batch_len = len(survtime)
    R_mat = np.zeros([current_batch_len, current_batch_len], dtype=int)
    #print(len(R_mat))
    for i in range(current_batch_len):
        for j in range(current_batch_len):
            R_mat[i,j] = survtime[j] >= survtime[i]

    R_mat = torch.FloatTensor(R_mat).to(device)
    theta = hazard_pred.reshape(-1)
    exp_theta = torch.exp(theta)
    #print(len(theta))
    loss_cox = -torch.mean((theta - torch.log(torch.sum(exp_theta*R_mat, dim=1))) * censor)
    return loss_cox


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def accuracy_cox(hazardsdata, labels):
    # This accuracy is based on estimated survival events against true survival events
    median = np.median(hazardsdata)
    hazards_dichotomize = np.zeros([len(hazardsdata)], dtype=int)
    hazards_dichotomize[hazardsdata > median] = 1
    correct = np.sum(hazards_dichotomize == labels)
    return correct / len(labels)


def cox_log_rank(hazardsdata, labels, survtime_all):
    median = np.median(hazardsdata)
    hazards_dichotomize = np.zeros([len(hazardsdata)], dtype=int)
    hazards_dichotomize[hazardsdata > median] = 1
    idx = hazards_dichotomize == 0
    T1 = survtime_all[idx]
    T2 = survtime_all[~idx]
    E1 = labels[idx]
    E2 = labels[~idx]
    results = logrank_test(T1, T2, event_observed_A=E1, event_observed_B=E2)
    pvalue_pred = results.p_value
    return(pvalue_pred)


def CIndex(hazards, labels, survtime_all):
    concord = 0.
    total = 0.
    N_test = labels.shape[0]
    for i in range(N_test):
        if labels[i] == 1:
            for j in range(N_test):
                if survtime_all[j] > survtime_all[i]:
                    total += 1
                    if hazards[j] < hazards[i]: concord += 1
                    elif hazards[j] < hazards[i]: concord += 0.5

    return(concord/total)


def CIndex_lifeline(hazards, labels, survtime_all):
    return(concordance_index(survtime_all, -hazards, labels))

def getCleanAllDataset_back(dataroot='./data/NSCLC/', ignore_missing_moltype=False, ignore_missing_histype=False, use_rnaseq=False):
    metadata = ['Histology','Grade', 'grade', 'Molecular subtype', 'ID', 'censored', 'Survival months']
    all_dataset = pd.read_csv(os.path.join(dataroot, 'all_dataset.csv'),index_col=False)
    all_dataset.index = all_dataset['ID']
    cols = all_dataset.columns.tolist()
    all_dataset = all_dataset[cols]
    if use_rnaseq:
        dataexp= pd.read_csv(os.path.join(dataroot, 'expression.txt'), sep='\t',  index_col=0)
        dataexp = dataexp[dataexp.columns[~dataexp.isnull().all()]]
        dataexp = dataexp.dropna(axis=1)
        dataexp.columns = [gene+'_rnaseq' for gene in dataexp.columns]
        dataexp.index = [patname[:12] for patname in dataexp.index]
        dataexp = dataexp.iloc[~dataexp.index.duplicated()]
        dataexp.index.name = 'ID'
        all_dataset = all_dataset.join(dataexp, how='inner')
    pat_missing_Grade =  all_dataset[all_dataset['Grade'].isna()].index
    pat_missing_histype = all_dataset[all_dataset['Histology'].isna()].index
    print("# Missing Histological Subtype:", len(pat_missing_histype))
    print("# Missing Grade:", len(pat_missing_Grade))
    assert pat_missing_histype.equals(pat_missing_Grade)
    ### 2. Impute Missing Genomic Data: Removes patients with missing molecular subtype / idh mutation / 1p19q. Else imputes with median value of each column. Fills missing Molecular subtype with "Missing"
    if ignore_missing_moltype: 
        all_dataset = all_dataset[all_dataset['Molecular subtype'].isna() == False]
    for col in all_dataset.drop(metadata, axis=1).columns:
        all_dataset['Molecular subtype'] = all_dataset['Molecular subtype'].fillna('Missing')
        all_dataset[col] = all_dataset[col].fillna(all_dataset[col].median())
    ### 3. Impute Missing Histological Data: Removes patients with missing histological subtype / Grade. Else imputes with "missing" / Grade -1
    if ignore_missing_histype: 
        all_dataset = all_dataset[all_dataset['Histology'].isna() == False]
    else:
        all_dataset['Grade'] = all_dataset['Grade'].fillna(1)
        all_dataset['Histology'] = all_dataset['Histology'].fillna('Missing')
    all_dataset['Grade'] = all_dataset['Grade'] 
    all_dataset['censored'] = all_dataset['censored']
    return metadata, all_dataset
    
    ### updatae 0406
def getCleanAllDataset(dataroot='/export/home/kongyan/project/data_fusion/data/BRCA/', ignore_missing_moltype=False, ignore_missing_histype=False, use_rnaseq=False):
    ### 1. Joining all_datasets.csv with Grade data. Looks at columns with misisng samples
    metadata = ['Histology','Grade', 'grade', 'Molecular subtype', 'ID', 'censored', 'Survival months']
    all_dataset1 = pd.read_csv(os.path.join(dataroot, 'all_dataset1.csv'),index_col=False)
    all_dataset1.index = all_dataset1['ID']
    cols = all_dataset1.columns.tolist()
    all_dataset1 = all_dataset1[cols]
    all_dataset2 = pd.read_csv(os.path.join(dataroot, 'all_dataset2.csv'),index_col=False)
    all_dataset2.index = all_dataset2['ID']
    cols = all_dataset2.columns.tolist()
    all_dataset2 = all_dataset2[cols]
    all_dataset3 = pd.read_csv(os.path.join(dataroot, 'all_dataset3.csv'),index_col=False)
    all_dataset3.index = all_dataset3['ID']
    cols = all_dataset3.columns.tolist()
    all_dataset3 = all_dataset3[cols]
    if use_rnaseq:
        dataexp= pd.read_csv(os.path.join(dataroot, 'expression.txt'), sep='\t',  index_col=0)
        datamut= pd.read_csv(os.path.join(dataroot, 'mutation.txt'), sep='\t',  index_col=0)
        datameth= pd.read_csv(os.path.join(dataroot, 'methylation.txt'), sep='\t',  index_col=0)
        #lgg = pd.read_csv(os.path.join(dataroot, 'mRNA_Expression_Zscores_RSEM.txt'), sep='\t', skiprows=1, index_col=0)
        dataexp = dataexp[dataexp.columns[~dataexp.isnull().all()]]
        datamut = datamut[datamut.columns[~datamut.isnull().all()]]
        datameth = datameth[datameth.columns[~datameth.isnull().all()]]
        #lgg = lgg[lgg.columns[~lgg.isnull().all()]]
        #glioma_RNAseq = gbm.join(lgg, how='inner').T
        dataexp = dataexp.dropna(axis=1)
        dataexp.columns = [gene+'_rnaseq' for gene in dataexp.columns]
        dataexp.index = [patname[:12] for patname in dataexp.index]
        dataexp = dataexp.iloc[~dataexp.index.duplicated()]
        dataexp.index.name = 'ID'
        all_dataset1 = all_dataset1.join(dataexp, how='inner')
        #
        datamut = datamut.dropna(axis=1)
        datamut.columns = [gene+'_mut' for gene in datamut.columns]
        datamut.index = [patname[:12] for patname in datamut.index]
        datamut = datamut.iloc[~datamut.index.duplicated()]
        datamut.index.name = 'ID'
        all_dataset2 = all_dataset2.join(datamut, how='inner')
        #
        datameth = datameth.dropna(axis=1)
        datameth.columns = [gene+'_meth' for gene in datameth.columns]
        datameth.index = [patname[:12] for patname in datameth.index]
        datameth = datameth.iloc[~datameth.index.duplicated()]
        datameth.index.name = 'ID'
        all_dataset3 = all_dataset3.join(datameth, how='inner')
        
        
        
        
    pat_missing_Grade =  all_dataset1[all_dataset1['Grade'].isna()].index
    pat_missing_histype = all_dataset1[all_dataset1['Histology'].isna()].index
   # print("# Missing Histological Subtype:", len(pat_missing_histype))
    pat_missing_Grade =  all_dataset2[all_dataset2['Grade'].isna()].index
    pat_missing_histype = all_dataset2[all_dataset2['Histology'].isna()].index
   # print("# Missing Histological Subtype:", len(pat_missing_histype))
    pat_missing_Grade =  all_dataset3[all_dataset3['Grade'].isna()].index
    pat_missing_histype = all_dataset3[all_dataset3['Histology'].isna()].index
   # print("# Missing Histological Subtype:", len(pat_missing_histype))
   # print("# Missing Grade:", len(pat_missing_Grade))
    assert pat_missing_histype.equals(pat_missing_Grade)
    ### 2. Impute Missing Genomic Data: Removes patients with missing molecular subtype / idh mutation / 1p19q. Else imputes with median value of each column. Fills missing Molecular subtype with "Missing"
    if ignore_missing_moltype: 
        all_dataset1 = all_dataset1[all_dataset1['Molecular subtype'].isna() == False]
        all_dataset2 = all_dataset2[all_dataset2['Molecular subtype'].isna() == False]
        all_dataset3 = all_dataset3[all_dataset3['Molecular subtype'].isna() == False]
    for col in all_dataset1.drop(metadata, axis=1).columns:
        all_dataset1['Molecular subtype'] = all_dataset1['Molecular subtype'].fillna('Missing')
        all_dataset1[col] = all_dataset1[col].fillna(all_dataset1[col].median())
    for col in all_dataset2.drop(metadata, axis=1).columns:
        all_dataset2['Molecular subtype'] = all_dataset2['Molecular subtype'].fillna('Missing')
        all_dataset2[col] = all_dataset2[col].fillna(all_dataset2[col].median())
    for col in all_dataset3.drop(metadata, axis=1).columns:
        all_dataset3['Molecular subtype'] = all_dataset3['Molecular subtype'].fillna('Missing')
        all_dataset3[col] = all_dataset3[col].fillna(all_dataset3[col].median())

    ### 3. Impute Missing Histological Data: Removes patients with missing histological subtype / Grade. Else imputes with "missing" / Grade -1
    if ignore_missing_histype: 
        all_dataset1 = all_dataset1[all_dataset1['Histology'].isna() == False]
        all_dataset2 = all_dataset2[all_dataset2['Histology'].isna() == False]
        all_dataset3 = all_dataset3[all_dataset3['Histology'].isna() == False]
    else:
        all_dataset1['Grade'] = all_dataset1['Grade'].fillna(1)
        all_dataset2['Grade'] = all_dataset2['Grade'].fillna(1)
        all_dataset3['Grade'] = all_dataset3['Grade'].fillna(1)
        all_dataset1['Histology'] = all_dataset1['Histology'].fillna('Missing')
        all_dataset2['Histology'] = all_dataset2['Histology'].fillna('Missing')
        all_dataset3['Histology'] = all_dataset3['Histology'].fillna('Missing')

#    all_dataset1['Grade'] = all_dataset1['Grade'] 
    all_dataset1['Grade'] = all_dataset1['Grade'] 
    all_dataset1['censored'] = all_dataset1['censored']
    all_dataset2['Grade'] = all_dataset2['Grade'] 
    all_dataset2['censored'] = all_dataset2['censored']
    all_dataset3['Grade'] = all_dataset3['Grade'] 
    all_dataset3['censored'] = all_dataset3['censored']
    return metadata, all_dataset1,all_dataset2,all_dataset3


################
# Analysis Utils
################
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def hazard2Grade(hazard, p):
    if hazard < p[0]:
        return 0
    elif hazard < p[1]:
        return 1
    return 2


def p(n):
    def percentile_(x):
        return np.percentile(x, n)
    percentile_.__name__ = 'p%s' % n
    return percentile_


def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)


def CI_pm(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return str("{0:.4f} ± ".format(m) + "{0:.3f}".format(h))


def CI_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return str("{0:.3f}, ".format(m-h) + "{0:.3f}".format(m+h))


def poolSurvTestPD(ckpt_name='./checkpoints/TCGA_GBMLGG/surv_15_rnaseq/', model='pathgraphomic_fusion', split='test', zscore=False, agg_type='Hazard_mean'):
    all_dataset_regstrd_pooled = []    
    ignore_missing_moltype = 1
    #ignore_missing_moltype = 1 if 'omic' in model else 0
    ignore_missing_histype = 1
    #ignore_missing_histype = 1 if 'grad' in ckpt_name else 0
    use_patch, roi_dir, use_vgg_features = ('_patch_', 'patches_512', 1) if ((('path' in model) or ('graph' in model)) and ('cox' not in model)) else ('_', 'Image', 0)
    use_patch='_'
    #use_patch, roi_dir, use_vgg_features = ('_patch_', 'all_st_patches_512', 1) if ((('path' in model) or ('graph' in model)) and ('cox' not in model)) else ('_', 'all_st', 0)
    #use_rnaseq = '_rnaseq' if ('rnaseq' in ckpt_name and 'path' != model and 'pathpath' not in model and 'graph' != model and 'graphgraph' not in model) else ''
    use_rnaseq = '_rnaseq' 

    for k in range(0,10):
        if k ==0:
            pred = pickle.load(open(ckpt_name+'%s/%s_%d%spred_%s.pkl' % (model, model, k, use_patch, split), 'rb'))    
            if 'grad' not in ckpt_name:
                surv_all = pd.DataFrame(np.stack(np.delete(np.array(pred), 3))).T
                surv_all.columns = ['Hazard', 'Survival months', 'censored', 'Grade']
                data_cv = pickle.load(open('./data/NSCLC/splits/data10cv_3omic_%s_%d_%d_%d%s.pkl' % (roi_dir, ignore_missing_moltype, ignore_missing_histype, use_vgg_features, use_rnaseq), 'rb'))
                data_cv_splits = data_cv['cv_splits']
                data_cv_split_k = data_cv_splits[k]
                all_dataset = data_cv['data_pd'].drop('ID', axis=1)
                all_dataset_regstrd = all_dataset.loc[data_cv_split_k[split]['x_patname']] # Subset of "all_datasets" (metadata) that is registered with "pred" (predictions)
                all_dataset_regstrd.insert(loc=0, column='Hazard', value = np.array(surv_all['Hazard']))
                all_dataset_regstrd.index.name = 'ID'
                hazard_agg = all_dataset_regstrd.groupby('ID').agg({'Hazard': ['mean', 'median', max, p(0.25), p(0.75)]})
                hazard_agg.columns = ["_".join(x) for x in hazard_agg.columns.ravel()]
                hazard_agg = hazard_agg[[agg_type]]
                hazard_agg.columns = ['Hazard']
                pred = hazard_agg.join(all_dataset, how='inner')
        if zscore: pred['Hazard'] = scipy.stats.zscore(np.array(pred['Hazard']))
        all_dataset_regstrd_pooled.append(pred)

    all_dataset_regstrd_pooled = pd.concat(all_dataset_regstrd_pooled)
    return all_dataset_regstrd_pooled

def makeKaplanMeierPlot(ckpt_name='./checkpoints/surv/', model='omic', split='test', zscore=False, agg_type='Hazard_mean'):
    def hazard2KMCurve(data, subtype):
        p = np.percentile(data['Hazard'], [33, 66])
        if p[0] == p[1]: p[0] = 2.99997
        data.insert(0, 'Grade_pred', [hazard2Grade(hazard, p) for hazard in data['Hazard']])
        kmf_pred = lifelines.KaplanMeierFitter()
        kmf_gt = lifelines.KaplanMeierFitter()

        def get_name(model):
            mode2name = { 'graph':'Histology GCN', 'omic':'Genomic SNN','graphomic':'Pathomic F,'}
            #mode2name = {'pathgraphomic':'Pathomic F.', 'pathomic':'Pathomic F.', 'graphomic':'Pathomic F.', 'path':'Histology CNN', 'graph':'Histology GCN', 'omic':'Genomic SNN'}
            for mode in mode2name.keys():
                if mode in model: return mode2name[mode]
            return 'N/A'

        fig = plt.figure(figsize=(10, 10), dpi=600)
        ax = plt.subplot()
        censor_style = {'ms': 20, 'marker': '+'}
        
        temp = data[data['Grade']==0]
        kmf_gt.fit(temp['Survival months']/365, temp['censored'], label="Grade II")
        kmf_gt.plot(ax=ax, show_censors=True, ci_show=False, c='g', linewidth=3, ls='--', markerfacecolor='black', censor_styles=censor_style)
        temp = data[data['Grade_pred']==0]
        kmf_pred.fit(temp['Survival months']/365, temp['censored'], label="%s (Low)" % get_name(model))
        kmf_pred.plot(ax=ax, show_censors=True, ci_show=False, c='g', linewidth=4, ls='-', markerfacecolor='black', censor_styles=censor_style)

        temp = data[data['Grade']==1]
        kmf_gt.fit(temp['Survival months']/365, temp['censored'], label="Grade III")
        kmf_gt.plot(ax=ax, show_censors=True, ci_show=False, c='b', linewidth=3, ls='--', censor_styles=censor_style)
        temp = data[data['Grade_pred']==1]
        kmf_pred.fit(temp['Survival months']/365, temp['censored'], label="%s (Mid)" % get_name(model))
        kmf_pred.plot(ax=ax, show_censors=True, ci_show=False, c='b', linewidth=4, ls='-', censor_styles=censor_style)

        if subtype != 'ODG':    
            temp = data[data['Grade']==2]
            kmf_gt.fit(temp['Survival months']/365, temp['censored'], label="Grade IV")
            kmf_gt.plot(ax=ax, show_censors=True, ci_show=False, c='r', linewidth=3, ls='--', censor_styles=censor_style)
            temp = data[data['Grade_pred']==2]
            kmf_pred.fit(temp['Survival months']/365, temp['censored'], label="%s (High)" % get_name(model))
            kmf_pred.plot(ax=ax, show_censors=True, ci_show=False, c='r', linewidth=4, ls='-', censor_styles=censor_style)

        ax.set_xlabel('')
        ax.set_ylim(0, 1)
        ax.set_yticks(np.arange(0, 1.001, 0.5))

        ax.tick_params(axis='both', which='major', labelsize=40)    
        plt.legend(fontsize=32, prop=font_manager.FontProperties(family='Arial', style='normal', size=32))
        if subtype != 'idhwt_ATC': ax.get_legend().remove()
        return fig
    
    data = poolSurvTestPD(ckpt_name, model, split, zscore, agg_type)
   # for subtype in ['idhwt_ATC', 'idhmut_ATC', 'ODG']:
    #    fig = hazard2KMCurve(data[data['Histomolecular subtype'] == subtype], subtype)
     #   fig.savefig(ckpt_name+'/%s_KM_%s.png' % (model, subtype))
        
    fig = hazard2KMCurve(data, 'all')
    fig.savefig(ckpt_name+'/%s_KM_%s.png' % (model, 'all'))



def makeAUROCPlot(ckpt_name='./checkpoints/grad/', model_list=['path', 'omic', 'pathgraphomic_fusion'], split='test', avg='micro', use_zoom=False):
    mpl.rcParams['font.family'] = "arial"
    colors = {'path':'dodgerblue', 'graph':'orange', 'omic':'green', 'pathgraphomic_fusion':'crimson'}
    names = {'path':'Histology CNN', 'graph':'Histology GCN', 'omic':'Genomic SNN', 'pathgraphomic_fusion':'Pathomic F.'}
    zoom_params = {0:([0.2, 0.4], [0.8, 1.0]), 
                   1:([0.25, 0.45], [0.75, 0.95]),
                   2:([0.0, 0.2], [0.8, 1.0]),
                   'micro':([0.15, 0.35], [0.8, 1.0])}
    mean_fpr = np.linspace(0, 1, 100)
    classes = [0, 1, 2, avg]
    ### 1. Looping over classes
    for i in classes:
        print("Class: " + str(i))
        fi = pylab.figure(figsize=(10,10), dpi=600, linewidth=0.2)
        axi = plt.subplot()
        
        ### 2. Looping over models
        for m, model in enumerate(model_list):
            ignore_missing_moltype = 1 if 'omic' in model else 0
            ignore_missing_histype = 1 if 'grad' in ckpt_name else 0
            use_patch, roi_dir, use_vgg_features = ('_patch_', 'all_st_patches_512', 1) if (('path' in model) or ('graph' in model)) else ('_', 'Image', 0)
            #use_patch, roi_dir, use_vgg_features = ('_patch_', 'all_st_patches_512', 1) if (('path' in model) or ('graph' in model)) else ('_', 'all_st', 0)

            ###. 3. Looping over all splits
            tprs, pres, aucrocs, rocaucs, = [], [], [], []
            for k in range(1,11):
                pred = pickle.load(open(ckpt_name+'/%s/%s_%d%spred_%s.pkl' % (model, model, k, use_patch, split), 'rb'))    
                Grade_pred, Grade = np.array(pred[3]), np.array(pred[4])
                enc = LabelBinarizer()
                enc.fit(Grade)
                Grade_oh = enc.transform(Grade)

                if i != avg:
                    pres.append(average_precision_score(Grade_oh[:, i], Grade_pred[:, i])) # from https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html
                    fpr, tpr, thresh = roc_curve(Grade_oh[:,i], Grade_pred[:,i], drop_intermediate=False)
                    aucrocs.append(auc(fpr, tpr)) # https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
                    rocaucs.append(roc_auc_score(Grade_oh[:,i], Grade_pred[:,i])) # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score
                    tprs.append(interp(mean_fpr, fpr, tpr))
                    tprs[-1][0] = 0.0
                else:
                    # A "micro-average": quantifying score on all classes jointly
                    pres.append(average_precision_score(Grade_oh, Grade_pred, average=avg))
                    fpr, tpr, thresh = roc_curve(Grade_oh.ravel(), Grade_pred.ravel())
                    aucrocs.append(auc(fpr, tpr))
                    rocaucs.append(roc_auc_score(Grade_oh, Grade_pred, avg))
                    tprs.append(interp(mean_fpr, fpr, tpr))
                    tprs[-1][0] = 0.0

            mean_tpr = np.mean(tprs, axis=0)
            mean_tpr[-1] = 1.0
            #mean_auc = auc(mean_fpr, mean_tpr)
            mean_auc = np.mean(aucrocs)
            std_auc = np.std(aucrocs)
            print('\t'+'%s - AUC: %0.3f ± %0.3f' % (model, mean_auc, std_auc))
            
            if use_zoom:
                alpha, lw = (0.8, 6) if model =='pathgraphomic_fusion' else (0.5, 6)
                plt.plot(mean_fpr, mean_tpr, color=colors[model],
                     label=r'%s (AUC = %0.3f $\pm$ %0.3f)' % (names[model], mean_auc, std_auc), lw=lw, alpha=alpha)
                std_tpr = np.std(tprs, axis=0)
                tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
                tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
                plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color=colors[model], alpha=0.1)
                plt.xlim([zoom_params[i][0][0]-0.005, zoom_params[i][0][1]+0.005])
                plt.ylim([zoom_params[i][1][0]-0.005, zoom_params[i][1][1]+0.005])
                axi.set_xticks(np.arange(zoom_params[i][0][0], zoom_params[i][0][1]+0.001, 0.05))
                axi.set_yticks(np.arange(zoom_params[i][1][0], zoom_params[i][1][1]+0.001, 0.05))
                axi.tick_params(axis='both', which='major', labelsize=26)
            else:
                alpha, lw = (0.8, 4) if model =='pathgraphomic_fusion' else (0.5, 3)
                plt.plot(mean_fpr, mean_tpr, color=colors[model],
                     label=r'%s (AUC = %0.3f $\pm$ %0.3f)' % (names[model], mean_auc, std_auc), lw=lw, alpha=alpha)
                std_tpr = np.std(tprs, axis=0)
                tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
                tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
                plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color=colors[model], alpha=0.1)
                plt.xlim([-0.05, 1.05])
                plt.ylim([-0.05, 1.05])
                axi.set_xticks(np.arange(0, 1.001, 0.2))
                axi.set_yticks(np.arange(0, 1.001, 0.2))
                axi.legend(loc="lower right", prop={'size': 20})
                axi.tick_params(axis='both', which='major', labelsize=30)
                #plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='navy', alpha=.8)

    figures = [manager.canvas.figure
               for manager in mpl._pylab_helpers.Gcf.get_all_fig_managers()]
    
    zoom = '_zoom' if use_zoom else ''
    for i, fig in enumerate(figures):
        fig.savefig(ckpt_name+'/AUC_%s%s.png' % (classes[i], zoom))

flags.DEFINE_integer("seed", 42, "fixed seed to apply to all rng entrypoints")
FLAGS = flags.FLAGS
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def is_non_empty_file(path):
    if isinstance(path, str):
        path = Path(path)
    return path.is_file() and path.stat().st_size != 0
def set_seeds(seed=42, fully_deterministic=False):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if fully_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
def load_model(model_path, map_location=device):
    model_path = os.path.realpath(model_path)  # Resolve any symlinks
    return torch.load(model_path, map_location=map_location)
class FixedRandomState:
    def __init__(self, seed=0):
        self.seed = seed
    def __enter__(self):
        self.random_state = RandomStateCache()
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
    def __exit__(self, *args):
        self.random_state.restore()
class RandomStateCache:
    def __init__(self):
        self.store()
    def store(self):
        self.random_state = random.getstate()
        self.numpy_state = np.random.get_state()
        self.torch_state = torch.random.get_rng_state()
        if torch.cuda.is_available():
            self.cuda_state = torch.cuda.get_rng_state_all()
    def restore(self):
        random.setstate(self.random_state)
        np.random.set_state(self.numpy_state)
        torch.random.set_rng_state(self.torch_state)
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(self.cuda_state)
class RAdam(Optimizer):
    def __init__(
        self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, degenerated_to_sgd=True
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        self.degenerated_to_sgd = degenerated_to_sgd
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

        super(RAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RAdam, self).__setstate__(state)

    def reset_step_buffer(self):
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["step"] = 0

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.reset_step_buffer()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError("RAdam does not support sparse gradients")

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p_data_fp32)
                    state["exp_avg_sq"] = torch.zeros_like(p_data_fp32)
                else:
                    state["exp_avg"] = state["exp_avg"].type_as(p_data_fp32)
                    state["exp_avg_sq"] = state["exp_avg_sq"].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                state["step"] += 1
                beta2_t = beta2 ** state["step"]
                N_sma_max = 2 / (1 - beta2) - 1
                N_sma = N_sma_max - 2 * state["step"] * beta2_t / (1 - beta2_t)

                # more conservative since it's an approximated value
                if N_sma >= 5:
                    if group["weight_decay"] != 0:
                        p_data_fp32.add_(-group["weight_decay"] * group["lr"], p_data_fp32)
                    step_size = (
                        group["lr"]
                        * math.sqrt(
                            (1 - beta2_t)
                            * (N_sma - 4)
                            / (N_sma_max - 4)
                            * (N_sma - 2)
                            / N_sma
                            * N_sma_max
                            / (N_sma_max - 2)
                        )
                        / (1 - beta1 ** state["step"])
                    )
                    denom = exp_avg_sq.sqrt().add_(group["eps"])
                    p_data_fp32.addcdiv_(-step_size, exp_avg, denom)
                    p.data.copy_(p_data_fp32)
                elif self.degenerated_to_sgd:
                    if group["weight_decay"] != 0:
                        p_data_fp32.add_(-group["weight_decay"] * group["lr"], p_data_fp32)
                    step_size = group["lr"] / (1 - beta1 ** state["step"])
                    p_data_fp32.add_(-step_size, exp_avg)
                    p.data.copy_(p_data_fp32)

        return loss
class BatchNorm(torch.nn.Module):
    def __init__(self, num_features, batch_norm_on):
        super().__init__()

        self.num_features = num_features
        self.batch_norm_on = batch_norm_on

        if batch_norm_on:
            self.bn = torch.nn.BatchNorm1d(num_features)
        else:
            self.bn = torch.nn.Identity()

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.bn(x)
        x = x.transpose(1, 2)
        return x
class Permute(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(0, 2, 1)


class GlobalNormalization(torch.nn.Module):
    """
    """
    def __init__(self, feature_dim, scale=False):
        super().__init__()
        self.feature_dim = feature_dim
        self.register_buffer("running_ave", torch.zeros(1, 1, self.feature_dim))
        self.register_buffer("total_frames_seen", torch.Tensor([0]))
        self.scale = scale
        self.register_buffer("running_sq_diff", torch.zeros(1, 1, self.feature_dim))

    def forward(self, inputs):
        if len(inputs.shape) != 3 or inputs.shape[2] != self.feature_dim:
            raise ValueError(
                f"""Inputs do not match required shape [batch_size, window_size, feature_dim], """
                f"""(expecting feature dim {self.feature_dim}), got {inputs.shape}"""
            )
        if self.training:
            self.update_stats(inputs)

        if self.scale:
            std = torch.sqrt(self.running_sq_diff / self.total_frames_seen)
            inputs = (inputs - self.running_ave) / std
        else:
            inputs = inputs - self.running_ave

        return inputs

    def unnorm(self, inputs):
        if self.scale:
            std = torch.sqrt(self.running_sq_diff / self.total_frames_seen)
            inputs = inputs * std + self.running_ave
        else:
            inputs = inputs + self.running_ave

        return inputs

    def update_stats(self, inputs):
        inputs_for_stats = inputs.detach()
        frames_in_input = inputs.shape[0] * inputs.shape[1]
        updated_running_ave = (
            self.running_ave * self.total_frames_seen
            + inputs_for_stats.sum(dim=(0, 1), keepdim=True)
        ) / (self.total_frames_seen + frames_in_input)

        if self.scale:
            # Update the sum of the squared differences between inputs and mean
            self.running_sq_diff = self.running_sq_diff + (
                (inputs_for_stats - self.running_ave) * (inputs_for_stats - updated_running_ave)
            ).sum(dim=(0, 1), keepdim=True)

        self.running_ave = updated_running_ave
        self.total_frames_seen = self.total_frames_seen + frames_in_input


def wav_to_float(x):
    """
    """
    assert x.dtype == torch.int16, f"got {x.dtype}"
    max_value = torch.iinfo(torch.int16).max
    min_value = torch.iinfo(torch.int16).min
    if not x.is_floating_point():
        x = x.to(torch.float)
    x = x - min_value
    x = x / ((max_value - min_value) / 2.0)
    x = x - 1.0
    return x
def float_to_wav(x):
    """
    """
    assert x.dtype == torch.float
    max_value = torch.iinfo(torch.int16).max
    min_value = torch.iinfo(torch.int16).min

    x = x + 1.0
    x = x * (max_value - min_value) / 2.0
    x = x + min_value
    x = x.to(torch.int16)
    return x
def mu_law_encoding(x, mu=255.0):
    """
    Input in range -2**15, 2*15 (or what is determined from dtype)
    Output is in range -1, 1 on mu law scale
    """
    x = wav_to_float(x)
    mu = torch.tensor(mu, dtype=x.dtype, device=x.device)
    x_mu = torch.sign(x) * (torch.log1p(mu * torch.abs(x)) / torch.log1p(mu))
    return x_mu

def fig2tensor(fig):
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    x = np.array(img)
    x = torch.Tensor(x).permute(2, 0, 1) / 255.0
    return x
def get_cell_image(img, cx, cy, size=512):
    cx = 32 if cx < 32 else size-32 if cx > size-32 else cx
    cy = 32 if cy < 32 else size-32 if cy > size-32 else cy
    return img[cy-32:cy+32, cx-32:cx+32, :]

def get_cpc_features(cell):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    cell = transform(cell)
    cell = cell.unsqueeze(0)
    device = torch.device('cuda:{}'.format('0'))
    feats = encoder(cell.to(device)).cpu().detach().numpy()[0]
    return feats
def get_cell_features(img, contour):  
    # Get contour coordinates from contour
    (cx, cy), (short_axis, long_axis), angle = cv2.fitEllipse(contour)
    cx, cy = int(cx), int(cy)   
    # Get a 64 x 64 center crop over each cell    
    img_cell = get_cell_image(img, cx, cy)
    grey_region = cv2.cvtColor(img_cell, cv2.COLOR_RGB2GRAY)
    img_cell_grey = np.pad(grey_region, [(0, 64-grey_region.shape[0]), (0, 64-grey_region.shape[1])], mode = 'reflect') 
    # 1. Generating contour features
    eccentricity = math.sqrt(1-(short_axis/long_axis)**2)
    convex_hull = cv2.convexHull(contour)
    area, hull_area = cv2.contourArea(contour), cv2.contourArea(convex_hull)
    solidity = float(area)/hull_area
    arc_length = cv2.arcLength(contour, True)
    roundness = (arc_length/(2*math.pi))/(math.sqrt(area/math.pi))   
    # 2. Generating GLCM features
    out_matrix = skimage.feature.greycomatrix(img_cell_grey, [1], [0])
    dissimilarity = skimage.feature.greycoprops(out_matrix, 'dissimilarity')[0][0]
    homogeneity = skimage.feature.greycoprops(out_matrix, 'homogeneity')[0][0]
    energy = skimage.feature.greycoprops(out_matrix, 'energy')[0][0]
    ASM = skimage.feature.greycoprops(out_matrix, 'ASM')[0][0]  
    # 3. Generating CPC features
    cpc_feats = get_cpc_features(img_cell)
    # Concatenate + Return all features
    x = [[short_axis, long_axis, angle, area, arc_length, eccentricity, roundness, solidity],
         [dissimilarity, homogeneity, energy, ASM], 
         cpc_feats]   
    return np.array(list(itertools.chain(*x)), dtype=np.float64), cx, cy


def seg2graph(img, contours):
    G = nx.Graph()   
    contours = [c for c in contours if c.shape[0] > 5]
    for v, contour in enumerate(contours):
        features, cx, cy = get_cell_features(img, contour)
        G.add_node(v, centroid = [cx, cy], x = features)
    if v < 5: return None
    return G
def from_networkx_backup(G):
    r"""Converts a :obj:`networkx.Graph` or :obj:`networkx.DiGraph` to a
    :class:`torch_geometric.data.Data` instance.
    Args:
        G (networkx.Graph or networkx.DiGraph): A networkx graph.
    """
    G = G.to_directed() if not nx.is_directed(G) else G
    edge_index = torch.tensor(list(G.edges)).t().contiguous()
    keys = []
    keys += list(list(G.nodes(data=True))[0][1].keys())
    keys += list(list(G.edges(data=True))[0][2].keys())
    data = {key: [] for key in keys}
    for _, feat_dict in G.nodes(data=True):
        for key, value in feat_dict.items():
            data[key].append(value)

    for _, _, feat_dict in G.edges(data=True):
        for key, value in feat_dict.items():
            data[key].append(value)

    for key, item in data.items():
        #data[key] = torch.tensor(item)
        print(key)
        data[key] = item

    data['edge_index'] = edge_index
    data = torch_geometric.data.Data.from_dict(data)
    data.num_nodes = G.number_of_nodes()
    return data
def get_cell_image_og(img, cx, cy):
    if cx < 32 and cy < 32:
        return img[0: cy+32, 0:cx+32, :]
    elif cx < 32:
        return img[cy-32: cy+32, 0:cx+32, :] 
    elif cy < 32:
        return img[0: cy+32, cx-32:cx+32, :]
    else:
        return img[cy-32: cy+32, cx-32:cx+32, :]
def my_transform(img):
    img = F.to_tensor(img)
    img = F.normalize(img, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    return img

def get_cpc_features_og(cell):
    cell_R = np.squeeze(cell[:, :, 2:3])
    cell_G = np.squeeze(cell[:, :, 1:2])
    cell_B = np.squeeze(cell[:, :, 0:1])
    cell_R = np.pad(cell_R, [(0, 64-cell_R.shape[0]), (0, 64-cell_R.shape[1])], mode = 'constant')
    cell_G = np.pad(cell_G, [(0, 64-cell_G.shape[0]), (0, 64-cell_G.shape[1])], mode = 'constant')
    cell_B = np.pad(cell_B, [(0, 64-cell_B.shape[0]), (0, 64-cell_B.shape[1])], mode = 'constant')
    cell = np.stack((cell_R, cell_B, cell_G))  
    cell = np.transpose(cell, (1, 2, 0))
    cell = my_transform(cell)
    cell = cell.unsqueeze(0) 
    device = torch.device('cuda:{}'.format('0'))    
    feats = encoder(cell.to(device)).cpu().detach().numpy()
    return feats
