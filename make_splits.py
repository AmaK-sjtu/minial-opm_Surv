### data_loaders.py
import argparse
import os
import pickle

import numpy as np
import pandas as pd
from PIL import Image
from sklearn import preprocessing

# Env
from networks import define_net
from utils import getCleanAllDataset
import torch
from torchvision import transforms
from options import parse_gpuids

### Initializes parser and data
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, default='/export/home/kongyan/project/data_fusion/data/NSCLC/', help="datasets")
    parser.add_argument('--roi_dir', type=str, default='Image')
    parser.add_argument('--graph_dir', type=str, default='NSCLC_st_cpc_blue')
    parser.add_argument('--graph_feat_type', type=str, default='pt_bi', help="graph features to use")
    parser.add_argument('--ignore_missing_moltype', type=int, default=1, help="Ignore data points with missing molecular subtype")
    parser.add_argument('--ignore_missing_histype', type=int, default=1, help="Ignore data points with missign histology subtype")
    parser.add_argument('--make_all_train', type=int, default=0)
    parser.add_argument('--use_vgg_features', type=int, default=0)
    parser.add_argument('--use_rnaseq', type=int, default=1)
    parser.add_argument('--checkpoints_dir', type=str, default='/export/home/kongyan/project/data_fusion/checkpoints/NSCLC/', help='models are saved here')
    parser.add_argument('--exp_name', type=str, default='surv_10_rnaseq', help='name of the project. It decides where to store samples and models')
    parser.add_argument('--gpu_ids', type=str, default='1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--mode', type=str, default='path', help='mode')
    #parser.add_argument('--mode', type=str, default='path', help='mode')
    parser.add_argument('--model_name', type=str, default='path', help='mode')
    parser.add_argument('--task', type=str, default='grad', help='surv | grad')
    #parser.add_argument('--task', type=str, default='surv', help='surv | grad')
    parser.add_argument('--act_type', type=str, default='Sigmoid', help='activation function')
    parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
    parser.add_argument('--label_dim', type=int, default=5, help='size of output')
    parser.add_argument('--batch_size', type=int, default=15, help="Number of batches to train/test for. Default: 256")
    parser.add_argument('--path_dim', type=int, default=32)
    parser.add_argument('--init_type', type=str, default='none', help='network initialization [normal | xavier | kaiming | orthogonal | max]. Max seems to work well')
    parser.add_argument('--dropout_rate', default=0.25, type=float, help='0 - 0.25. Increasing dropout_rate helps overfitting. Some people have gone as high as 0.5. You can try adding more regularization')
    opt = parser.parse_known_args()[0]
    opt = parse_gpuids(opt)
    return opt

opt = parse_args()
device=torch.device('cuda:1')
metadata, all_dataset1, all_dataset2, all_dataset3 = getCleanAllDataset(opt.dataroot, opt.ignore_missing_moltype, opt.ignore_missing_histype, opt.use_rnaseq)

### Creates a mapping from ID -> Image ROI
img_fnames = os.listdir(os.path.join(opt.dataroot, opt.roi_dir))
#img_fnames = os.listdir(os.path.join('/export/home/kongyan/project/data_fusion/data/NSCLC/','Image'))
pat2img = {}
for pat, img_fname in zip([img_fname[:12] for img_fname in img_fnames], img_fnames):
    if pat not in pat2img.keys(): pat2img[pat] = []
    pat2img[pat].append(img_fname)
# pat2img === TCGAID: repeat1 repeat2 ...
### Dictionary file containing split information
data_dict = {}
data_dict['data_pd'] = all_dataset1
#data_dict['pat2img'] = pat2img
#data_dict['img_fnames'] = img_fnames
cv_splits = {}

### Extracting K-Fold Splits
pnas_splits = pd.read_csv(opt.dataroot+'pnas_splits.csv')
pnas_splits.columns = ['ID']+[str(k) for k in range(1, 11)]
pnas_splits.index = pnas_splits['ID']
pnas_splits = pnas_splits.drop(['ID'], axis=1)

### get path_feats
def get_vgg_features(model, device, img_path):
    if model is None:
        return img_path
#        return Image.open(img_path).convert('RGB')
    else:
        x_path = Image.open(img_path).convert('RGB')
        normalize = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        x_path = torch.unsqueeze(normalize(x_path), dim=0)
        features, hazard = model(x_path=x_path.to(device))
        return features.cpu().detach().numpy()

### method for constructing aligned
def getAlignedMultimodalData(opt, model, device, all_dataset1,  all_dataset2, all_dataset3,pat_split, pat2img):
    x_patname, x_path, x_grph, x_omic,x_meth,x_mut, e, t, g = [], [], [], [], [], [], [],[],[]
    for pat_name in pat_split:
        if pat_name not in all_dataset1.index: continue
        for img_fname in pat2img[pat_name]:
            grph_fname = img_fname.rstrip('.png')+'.pt'
            assert grph_fname in os.listdir(os.path.join(opt.dataroot, opt.graph_dir, opt.graph_feat_type))
            assert all_dataset1[all_dataset1['ID'] == pat_name].shape[0] == 1
            #x_patname.append(img_fname)
            x_patname.append(pat_name)
            x_path.append(get_vgg_features(model, device, os.path.join(opt.dataroot, opt.roi_dir, img_fname)))
            x_grph.append(os.path.join(opt.dataroot, opt.graph_dir, opt.graph_feat_type, grph_fname))
            x_omic.append(np.array(all_dataset1[all_dataset1['ID'] == pat_name].drop(metadata, axis=1)))
            x_mut.append(np.array(all_dataset2[all_dataset2['ID'] == pat_name].drop(metadata, axis=1)))
            x_meth.append(np.array(all_dataset3[all_dataset3['ID'] == pat_name].drop(metadata, axis=1)))
            e.append(int(all_dataset1[all_dataset1['ID']==pat_name]['censored']))
            t.append(int(all_dataset1[all_dataset1['ID']==pat_name]['Survival months']))
            g.append(int(all_dataset1[all_dataset1['ID']==pat_name]['Grade']))
            #g.append(int(all_dataset1[all_dataset1['ID']==pat_name]['grade']))
    #return x_patname, x_patname, x_grph, x_omic, x_mut,x_meth,e, t, g
    return x_patname, x_path, x_grph, x_omic,x_mut,x_meth, e, t, g

for k in range(len(pnas_splits.columns)):
    print('Creating Split %s' % k)
    pat_train = pnas_splits.index[pnas_splits.iloc[:,k] == 'Train'] if opt.make_all_train == 0 else pnas_splits.index
    pat_test = pnas_splits.index[pnas_splits.iloc[:,k]  == 'Test'] 
    cv_splits[int(k)] = {}
    model = None
    train_x_patname, train_x_path, train_x_grph, train_x_omic,train_x_mut,train_x_meth, train_e, train_t, train_g = getAlignedMultimodalData(opt, model, device, all_dataset1, all_dataset2, all_dataset3, pat_train, pat2img)
    #print(train_x_path)
    test_x_patname, test_x_path, test_x_grph, test_x_omic, test_x_mut,test_x_meth,test_e, test_t, test_g = getAlignedMultimodalData(opt, model, device, all_dataset1, all_dataset2, all_dataset3,pat_test, pat2img)
    train_x_omic,train_x_mut,train_x_meth, train_e, train_t = np.array(train_x_omic).squeeze(axis=1),np.array(train_x_mut).squeeze(axis=1),np.array(train_x_meth).squeeze(axis=1),np.array(train_e, dtype=np.float64), np.array(train_t, dtype=np.float64)
    test_x_omic, test_x_mut,test_x_meth,test_e, test_t = np.array(test_x_omic).squeeze(axis=1),np.array(test_x_mut).squeeze(axis=1),np.array(test_x_meth).squeeze(axis=1), np.array(test_e, dtype=np.float64), np.array(test_t, dtype=np.float64)
    scaler = preprocessing.StandardScaler().fit(train_x_omic)
    train_x_omic = scaler.transform(train_x_omic)
    test_x_omic = scaler.transform(test_x_omic)
    scaler = preprocessing.StandardScaler().fit(train_x_mut)
    train_x_mut = scaler.transform(train_x_mut)
    test_x_mut = scaler.transform(test_x_mut)
    scaler = preprocessing.StandardScaler().fit(train_x_meth)
    train_x_meth = scaler.transform(train_x_meth)
    test_x_meth = scaler.transform(test_x_meth)


    train_data = {'x_patname': train_x_patname,
                  'x_path':np.array(train_x_path),
                  'x_grph':train_x_grph,
                  'x_omic':train_x_omic,
                  'x_mut':train_x_mut,
                  'x_meth':train_x_meth,  
                  'e':np.array(train_e, dtype=np.float64), 
                  't':np.array(train_t, dtype=np.float64),
                  'g':np.array(train_g, dtype=np.float64)}
    test_data = {'x_patname': test_x_patname,
                 'x_path':np.array(test_x_path),
                 'x_grph':test_x_grph,
                 'x_omic':test_x_omic,
                 'x_mut':test_x_mut,
                 'x_meth':test_x_meth,  
                 'e':np.array(test_e, dtype=np.float64),
                 't':np.array(test_t, dtype=np.float64),
                 'g':np.array(test_g, dtype=np.float64)}
    dataset = {'train':train_data, 'test':test_data}
    cv_splits[int(k)] = dataset
    if opt.make_all_train: break
    
data_dict['cv_splits'] = cv_splits

#pickle.dump(data_dict, open('%s/splits/brca10cv_3omic_path_%s_%d_%d_%d%s.pkl' % (opt.dataroot, opt.roi_dir, opt.ignore_missing_moltype, opt.ignore_missing_histype, opt.use_vgg_features, '_rnaseq' if opt.use_rnaseq else ''), 'wb'))
pickle.dump(data_dict, open('%s/splits/brca10cv_3omic_%s_%d_%d_%d%s.pkl' % (opt.dataroot, opt.roi_dir, opt.ignore_missing_moltype, opt.ignore_missing_histype, opt.use_vgg_features, '_rnaseq' if opt.use_rnaseq else ''), 'wb'))
