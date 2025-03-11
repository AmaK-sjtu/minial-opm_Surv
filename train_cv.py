import os
from networks import define_net, define_reg, define_optimizer, define_scheduler

import logging
import numpy as np
import random
import pickle

import torch

# Env
from data_loaders import *
from options import parse_args
from train_test import train, test


### 1. Initializes parser and device
opt = parse_args()
#device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cuda')
device = torch.device(opt.gpu_ids[0])
#device = torch.device('cuda:1')

print("Using device:", device)
if not os.path.exists(opt.checkpoints_dir): os.makedirs(opt.checkpoints_dir)
if not os.path.exists(os.path.join(opt.checkpoints_dir, opt.exp_name)): os.makedirs(os.path.join(opt.checkpoints_dir, opt.exp_name))
if not os.path.exists(os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name)): os.makedirs(os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name))

### 2. Initializes Data
ignore_missing_histype = 1
#ignore_missing_histype = 1 if 'grad' in opt.task else 0
ignore_missing_moltype = 1 
#ignore_missing_moltype = 1 if 'omic' in opt.mode else 0
use_patch, roi_dir = ('_patch_', 'all_st_patches_512') if opt.use_vgg_features else ('_', 'Image')
roi_dir='Image'
#use_patch, roi_dir = ('_patch_', 'all_st_patches_512') if opt.use_vgg_features else ('_', 'all_st')
use_rnaseq = '_rnaseq' if opt.use_rnaseq else ''

data_cv_path = '%s/splits/lung10cv_3omic_%s_%d_%d_%d%s.pkl' % (opt.dataroot, roi_dir, ignore_missing_moltype, ignore_missing_histype, opt.use_vgg_features, use_rnaseq)
#data_cv_path = '%s/splits/brca10cv_3omic_%s_%d_%d_%d%s.pkl' % (opt.dataroot, roi_dir, ignore_missing_moltype, ignore_missing_histype, opt.use_vgg_features, use_rnaseq)
#data_cv_path = '%s/splits/brca10cv_3omic_path_Image_1_1_0_rnaseq.pkl' % opt.dataroot
#data_cv_path = '%s/splits/brca10cv_%s_%d_%d_%d%s.pkl' % (opt.dataroot, roi_dir, ignore_missing_moltype, ignore_missing_histype, opt.use_vgg_features, use_rnaseq)
print(data_cv_path)
#data_cv_path = '%s/splits/brca10cv_%s_%d_%d_%d%s.pkl' % (opt.dataroot, roi_dir, ignore_missing_moltype, ignore_missing_histype, opt.use_vgg_features, use_rnaseq)
print("Loading %s" % data_cv_path)
data_cv = pickle.load(open(data_cv_path, 'rb'))
data_cv_splits = data_cv['cv_splits']
TP=[]
FP=[]
TN=[]
FN=[]
results = []

### 3. Sets-Up Main Loop
for k, data in data_cv_splits.items():
	#if k >6:
		print("*******************************************")
		print("************** SPLIT (%d/%d) **************" % (k, len(data_cv_splits.items())))
		print("*******************************************")
		if os.path.exists(os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name, '%s_%d_patch_pred_train.pkl' % (opt.model_name, k))):
			print("Train-Test Split already made.")
			continue
	
		model = define_net(opt, k)
	
		### 3.1 Trains Mo
		model, optimizer, metric_logger = train(opt, data, device, k,model)
	
#		model.eval()
		### 3.2 Evalutes Train + Test Error, and Saves Model
		loss_train, cindex_train, pvalue_train, surv_acc_train, grad_acc_train, pred_train,tp1,fp1,tn1,fn1,re = test(opt, model, data, 'train', device)
	#	loss_test, cindex_test, pvalue_test, surv_acc_test, grad_acc_test, pred_test = test(opt, model, data, 'train', device)
		loss_test, cindex_test, pvalue_test, surv_acc_test, grad_acc_test, pred_test,tp2,fp2,tn2,fn2,re = test(opt, model, data, 'test', device)
	
		if opt.task == 'surv':
			f=open(os.path.join(opt.checkpoints_dir, opt.exp_name,opt.mode,"training.record.txt"),'a+')
			f.writelines("SPLIT: "+str(k)+"[Final] Apply model to training set: C-Index: "+str(cindex_train)+" ,P-Value: "+str(pvalue_train)+"\n")
			print("[Final] Apply model to training set: C-Index: %.10f, P-Value: %.10e" % (cindex_train, pvalue_train))
			logging.info("[Final] Apply model to training set: C-Index: %.10f, P-Value: %.10e" % (cindex_train, pvalue_train))
			print("[Final] Apply model to testing set: C-Index: %.10f, P-Value: %.10e" % (cindex_test, pvalue_test))
			f.writelines("SPLIT: "+str(k)+"[Final] Apply model to test set: C-Index: "+str(cindex_test)+" ,P-Value: "+str(pvalue_test)+"\n")
			f.close()
			logging.info("[Final] Apply model to testing set: C-Index: %.10f, P-Value: %.10e" % (cindex_test, pvalue_test))
			results.append(cindex_test)
		elif opt.task == 'grad':	
			f=open(os.path.join(opt.checkpoints_dir, opt.exp_name,opt.mode,"training.record.txt"),'a+')
			f.writelines("SPLIT: "+str(k)+" [Final] Apply model to training set: Loss: "+str(loss_train)+" ACC: "+str(grad_acc_train)+"\n")
			print("[Final] Apply model to training set: Loss: %.10f, Acc: %.4f" % (loss_train, grad_acc_train))
			logging.info("[Final] Apply model to training set: Loss: %.10f, Acc: %.4f" % (loss_train, grad_acc_train))
			print("[Final] Apply model to testing set: Loss: %.10f, Acc: %.4f" % (loss_test, grad_acc_test))
			f.writelines("SPLIT: "+str(k)+" [Final] Apply model to testing set: Loss: "+str(loss_test)+" ACC: "+str(grad_acc_test)+"\n")
			f.writelines("Train Set TP, FP, TN, FN: "+str(tp1)+" " +str(fp1)+" " +str(tn1)+" "+str(fn1)+"\n")
			f.writelines("Test Set TP, FP, TN, FN: "+str(tp2)+" " +str(fp2)+" " +str(tn2)+" "+str(fn2)+"\n")
			logging.info("[Final] Apply model to testing set: Loss: %.10f, Acc: %.4f" % (loss_test, grad_acc_test))
			f.close()
			results.append(grad_acc_test)
			TP.append(tp2)
			FP.append(fp2)
			TN.append(tn2)
			FN.append(fn2)
	              
	
		### 3.3 Saves Model
		if len(opt.gpu_ids) > 0 and torch.cuda.is_available():
		    model_state_dict = model.module.cuda().state_dict()
		else:
		    model_state_dict = model.cuda().state_dict()
		
		torch.save({
			'split':k,
		    'opt': opt,
		    'epoch': opt.niter+opt.niter_decay,
		    'data': data,
		    'model_state_dict': model_state_dict,
		    'optimizer_state_dict': optimizer.state_dict(),
		    'metrics': metric_logger}, 
		    os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name, '%s_%d.pt' % (opt.model_name, k)))
	
		print()
	
		pickle.dump(pred_train, open(os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name, '%s_%d%spred_train.pkl' % (opt.model_name, k, use_patch)), 'wb'))
		pickle.dump(pred_test, open(os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name, '%s_%d%spred_test.pkl' % (opt.model_name, k, use_patch)), 'wb'))


f=open(os.path.join(opt.checkpoints_dir, opt.exp_name,opt.mode,"training.record.txt"),'a+')
f.writelines("Split Results: "+str(results)+"\n")
print('Split Results:', results)
f.writelines("Average: "+str(np.array(results).mean())+"\n")
f.writelines("TP: "+str(TP)+"\n")
f.writelines("FP: "+str(FP)+"\n")
f.writelines("TN: "+str(TN)+"\n")
f.writelines("FN: "+str(FN)+"\n")
print("Average:", np.array(results).mean())
pickle.dump(results, open(os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name, '%s_results.pkl' % opt.model_name), 'wb'))
f.close()
