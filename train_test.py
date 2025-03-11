import torch.nn as nn
from Focalloss import lossn
import random
from tqdm import tqdm
import numpy as np
import torch
np.set_printoptions(threshold=np.inf)
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import RandomSampler

from data_loaders import PathgraphomicDatasetLoader, PathgraphomicFastDatasetLoader
from networks import define_net, define_reg, define_optimizer, define_scheduler
from utils import unfreeze_unimodal, CoxLoss, CIndex_lifeline, cox_log_rank, accuracy_cox, mixed_collate, count_parameters
import pdb
import pickle
import os
from options import parse_args
opt = parse_args()
device = torch.device(opt.gpu_ids[0])
def train(opt, data, device, k,model):
    cudnn.deterministic = True
    torch.cuda.manual_seed_all(2019)
    torch.manual_seed(2019)
    random.seed(2019)
    optimizer = define_optimizer(opt, model)
    scheduler = define_scheduler(opt, optimizer)
    optimizer.zero_grad()
    print(model)
    print("Number of Trainable Parameters: %d" % count_parameters(model))
    print("Activation Type:", opt.act_type)
    print("Optimizer Type:", opt.optimizer_type)
    print("Regularization Type:", opt.reg_type)
    use_patch='_'
    custom_data_loader =PathgraphomicFastDatasetLoader(opt, data, split='train', mode=opt.mode)
    #train_loader = torch.utils.data.DataLoader(dataset=custom_data_loader, batch_size=opt.batch_size, shuffle=True, drop_last=True)
    train_loader = torch.utils.data.DataLoader(dataset=custom_data_loader, batch_size=opt.batch_size, shuffle=True, collate_fn=mixed_collate,drop_last=False)
    metric_logger = {'train':{'loss':[], 'pvalue':[], 'cindex':[], 'surv_acc':[], 'grad_acc':[]},
                      'test':{'loss':[], 'pvalue':[], 'cindex':[], 'surv_acc':[], 'grad_acc':[]}}
    #print(opt.epoch_count, opt.niter+opt.niter_decay+1)
    for epoch in range(24):
    #for epoch in range(opt.epoch_count):
        print("EPOCH:"+str(epoch))
        if opt.finetune == 1:
            unfreeze_unimodal(opt, model, epoch)
        model = model.to(device)
        risk_pred_all, censor_all, survtime_all = np.array([]), np.array([]), np.array([])    # Used for calculating the C-Index
        loss_epoch, grad_acc_epoch = 0, 0

        f=open(os.path.join(opt.checkpoints_dir, opt.exp_name,opt.mode,"training.record.txt"),'a+')
        f.writelines("Epoch: "+str(epoch)+"\n")
        model.train()
        for batch_idx,( x_path,x_path2, x_grph, x_omic,x_mut,x_meth, censor, survtime, grade) in enumerate(train_loader):
            ######## 'x_grph': Batch(batch=[4050], centroid=[16], edge_attr=[19924, 1], edge_index=[2, 19924], x=[16])
            #print(x_mut)
            censor = censor.to(device) if "surv" in opt.task else censor
            grade = grade.to(device) if "grad" in opt.task else grade
            #print(x_omic)
        #    print(x_mut)
            _, pred = model(x_path=x_path.to(device), x_grph=x_grph.to(device), x_omic=x_omic.to(device),x_mut=x_mut.to(device),x_meth=x_meth.to(device),opt=opt,training=True)
            loss_cox = CoxLoss(survtime, censor, pred, device) if opt.task == "surv" else 0
            loss_reg = define_reg(opt, model)
            loss_nll=F.nll_loss(pred,grade) if opt.task=="grad" else 0
         ##   print(pred)
         #   loss_nll=lossn(pred,opt,grade) if opt.task=="grad" else 0
    #        print(loss_nll)
            loss= opt.lambda_cox*loss_cox + opt.lambda_nll*loss_nll + opt.lambda_reg*loss_reg
            print(loss)
   #         b=opt.bvalue
            #flood=(loss-b).abs()+b
            optimizer.zero_grad()
            loss.backward()
           # flood.backward()
            optimizer.step()
            loss_epoch += loss.data.item()
            if opt.task == "surv":
                tmp1=pred.detach().cpu().numpy()
                risk_pred_all = np.concatenate((risk_pred_all,tmp1.reshape(-1)))   # Logging Information
                tmp1=censor.detach().cpu().numpy()
                censor_all = np.concatenate((censor_all, tmp1.reshape(-1)))   # Logging Information
                tmp1=survtime.detach().cpu().numpy()
                survtime_all = np.concatenate((survtime_all,tmp1.reshape(-1)))   # Logging Information
            elif opt.task == "grad":
                pred = pred.argmax(dim=1, keepdim=True)
                grad_acc_epoch += pred.eq(grade.view_as(pred)).sum().item()
#                print(pred.eq(grade.view_as(pred)).sum().item())
            if opt.verbose > 0 and opt.print_every > 0 and (batch_idx % opt.print_every == 0 or batch_idx+1 == len(train_loader)):
                 print("Epoch {:02d}/{:02d} Batch {:04d}/{:d}, Loss {:9.4f}".format(
                     epoch+1, opt.niter+opt.niter_decay, batch_idx+1, len(train_loader), loss.item()))
                 f.writelines("Epoch: "+str(epoch+1)+" LOSS: "+str(loss.item())+"\n")
        scheduler.step()
        loss_epoch /= len(train_loader)
        print("On Test")
        print(risk_pred_all,censor_all) 
        cindex_epoch = CIndex_lifeline(risk_pred_all, censor_all, survtime_all) if opt.task == 'surv' else None
         
        pvalue_epoch = cox_log_rank(risk_pred_all, censor_all, survtime_all)  if opt.task == 'surv' else None
        surv_acc_epoch = accuracy_cox(risk_pred_all, censor_all)  if opt.task == 'surv' else None
        grad_acc_epoch = grad_acc_epoch / len(train_loader.dataset) if opt.task == 'grad' else None
    #    loss_test, cindex_test, pvalue_test, surv_acc_test, grad_acc_test, pred_test = test(opt, model, data, 'train', device)
#        loss_test, cindex_test, pvalue_test, surv_acc_test, grad_acc_test, pred_test,tp,fp,tn,fn = test(opt, model, data, 'test', device)
        loss_test, cindex_test, pvalue_test, surv_acc_test, grad_acc_test, pred_test,tp,fp,tn,fn,re = test(opt, model, data, 'test', device,k)
 
        metric_logger['train']['loss'].append(loss_epoch)
        metric_logger['train']['cindex'].append(cindex_epoch)
        metric_logger['train']['pvalue'].append(pvalue_epoch)
        metric_logger['train']['surv_acc'].append(surv_acc_epoch)
        metric_logger['train']['grad_acc'].append(grad_acc_epoch)
 
        metric_logger['test']['loss'].append(loss_test)
        metric_logger['test']['cindex'].append(cindex_test)
        metric_logger['test']['pvalue'].append(pvalue_test)
        metric_logger['test']['surv_acc'].append(surv_acc_test)
        metric_logger['test']['grad_acc'].append(grad_acc_test)
 
        pickle.dump(pred_test, open(os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name, '%s_%d%s%d_pred_test.pkl' % (opt.model_name, k, use_patch, epoch)), 'wb'))
 
        if opt.verbose > 0:
            if opt.task == 'surv':
                print('[{:s}]\t\tLoss: {:.4f}, {:s}: {:.4f}'.format('Train', loss_epoch, 'C-Index', cindex_epoch))
                f.writelines("Train loss "+str(loss_epoch)+" C-Index: "+str(cindex_epoch)+"\n")
 
                print('[{:s}]\t\tLoss: {:.4f}, {:s}: {:.4f}\n'.format('Test', loss_test, 'C-Index', cindex_test))
                f.writelines("Test loss: "+str(loss_test)+" C-Index: "+str(cindex_test)+"\n")
            elif opt.task == 'grad':
                print('[{:s}]\t\tLoss: {:.4f}, {:s}: {:.4f}'.format('Train', loss_epoch, 'Accuracy', grad_acc_epoch))
                f.writelines("Train loss: "+str(loss_epoch)+" Acc: "+str(grad_acc_epoch)+"\n")
                print('[{:s}]\t\tLoss: {:.4f}, {:s}: {:.4f}\n'.format('Test', loss_test, 'Accuracy', grad_acc_test))
                f.writelines("Test loss: "+str(loss_test)+" Acc: "+str(grad_acc_test)+"\n")
                f.writelines("TP, FP, TN, FN: "+str(tp)+" "+str(fp)+" "+str(tn)+" " +str(fn)+"\n")
 
        f.close()
    
    return model, optimizer, metric_logger


def test(opt, model, data, split, device, splitk):
    model.eval()

    crit=torch.nn.BCELoss()
    custom_data_loader = PathgraphomicFastDatasetLoader(opt, data, split, mode=opt.mode) 
    #custom_data_loader = PathgraphomicFastDatasetLoader(opt, data, split, mode=opt.mode) if opt.use_vgg_features else PathgraphomicDatasetLoader(opt, data, split=split, mode=opt.mode)
    test_loader = torch.utils.data.DataLoader(dataset=custom_data_loader, batch_size=opt.batch_size, shuffle=False, collate_fn=mixed_collate,drop_last=False)
#    print(len(test_loader))
    
    probs_all, gt_all = None, np.array([])
    loss_test, grad_acc_test = 0, 0
    tp,fp,fn,tn=0,0,0,0
    patname=np.array([])

    for batch_idx, (x_path,x_path2, x_grph, x_omic,x_mut,x_meth, censor, survtime, grade) in enumerate(test_loader):

        risk_pred_all, censor_all, survtime_all = np.array([]), np.array([]), np.array([])
        censor = censor.to(device) if "surv" in opt.task else censor
        grade = grade.to(device) if "grad" in opt.task else grade
        _, pred = model(x_path=x_path.to(device), x_grph=x_grph.to(device), x_omic=x_omic.to(device),x_mut=x_mut.to(device),x_meth=x_meth.to(device),training=False,dropout_rate=0,opt=opt)
        #print(grade)
        #print(pred)
        #print(x_mut)
        loss_cox = CoxLoss(survtime, censor, pred, device) if opt.task == "surv" else 0
        loss_reg = define_reg(opt, model)
     #   loss_nll = F.nll_loss(pred, grade) if opt.task == "grad" else 0
        loss_nll=lossn(pred,opt,grade) if opt.task=="grad" else 0
        loss = opt.lambda_cox*loss_cox + opt.lambda_nll*loss_nll + opt.lambda_reg*loss_reg
        loss_test += loss.data.item()
        print("Test loss:"+str(loss))
        tmp1=grade.cpu().numpy()
        gt_all = np.concatenate((gt_all,tmp1.reshape(-1)))   # Logging Information

        re=[]
        if opt.task == "surv":
            #x_path=x_path2.to(device)
            #tmp1=x_path2.detach().cpu().numpy()
            #names=  tmp1.reshape(-1)   # Logging Information
            #print(x_path2)
            f=open(str(splitk)+"0",'a+')
            #f.writelines(str(names)+"\n")
            f.writelines(str(x_path2)+"\n")
            f.close()
            tmp1=pred.detach().cpu().numpy()
            risk_pred_all = np.concatenate((risk_pred_all, tmp1.reshape(-1)))   # Logging Information
#            print(survtime)
            print(len(risk_pred_all))
            f=open(str(splitk)+"1",'a+')
            f.writelines(str(risk_pred_all)+"\n")
            f.close()
        #    print(risk_pred_all)
            tmp1=censor.detach().cpu().numpy()
            censor_all = np.concatenate((censor_all, tmp1.reshape(-1)))   # Logging Information
            f=open(str(splitk)+"2",'a+')
            f.writelines(str(censor_all)+"\n")
            f.close()
      #      print(censor_all)
            tmp1=survtime.detach().cpu().numpy()
            survtime_all = np.concatenate((survtime_all, tmp1.reshape(-1)))   # Logging Information
            f=open(str(splitk)+"3",'a+')
            f.writelines(str(survtime_all)+"\n")
            f.close()
     #       print(survtime_all)
            re=np.array([risk_pred_all,censor_all,survtime_all])
        elif opt.task == "grad":
            grade_pred = pred.argmax(dim=1, keepdim=True)
            grad_acc_test += grade_pred.eq(grade.view_as(grade_pred)).sum().item()
#            print(grade_pred.eq(grade.view_as(grade_pred)).sum().item())
            probs_np = pred.detach().cpu().numpy()
            probs_all = probs_np if probs_all is None else np.concatenate((probs_all, probs_np), axis=0)   # Logging Information
            tp +=(grade_pred*grade.view_as(grade_pred)).sum().item()
            fp +=(grade_pred*(grade_pred-grade.view_as(grade_pred))).sum().item()
            fn +=(grade.view_as(grade_pred)*(grade.view_as(grade_pred)-grade_pred)).sum().item()
            tn +=((1-grade_pred)*(1-grade.view_as(grade_pred))).sum().item()
    

    ################################################### 
    # ==== Measuring Test Loss, C-Index, P-Value ==== #
    ###################################################
    #print("TEST LOADER LENGTH:")
#    print(len(test_loader))
    loss_test /=len(test_loader)
    cindex_test = CIndex_lifeline(risk_pred_all, censor_all, survtime_all) if opt.task == 'surv' else None
    pvalue_test = cox_log_rank(risk_pred_all, censor_all, survtime_all) if opt.task == 'surv' else None
    surv_acc_test = accuracy_cox(risk_pred_all, censor_all) if opt.task == 'surv' else None
    grad_acc_test = grad_acc_test / len(test_loader.dataset) if opt.task == 'grad' else None
    pred_test = [risk_pred_all, survtime_all, censor_all, probs_all, gt_all]
#    print("TP, FP, TN, FN:"+str(tp)+" "+str(fp)+" "+str(tn)+" "+str(fn))

    #return loss_test, cindex_test, pvalue_test, surv_acc_test, grad_acc_test, pred_test,tp,fp,tn,fn
    return loss_test, cindex_test, pvalue_test, surv_acc_test, grad_acc_test, pred_test,tp,fp,tn,fn,re

def test2(opt, model, data, split, device):
    model.eval()

    crit=torch.nn.BCELoss()
    custom_data_loader = PathgraphomicFastDatasetLoader(opt, data, split, mode=opt.mode) if opt.use_vgg_features else PathgraphomicDatasetLoader(opt, data, split=split, mode=opt.mode)
    test_loader = torch.utils.data.DataLoader(dataset=custom_data_loader, batch_size=opt.batch_size, shuffle=True, collate_fn=mixed_collate,drop_last=False)
#    print(len(test_loader))
    
    risk_pred_all, censor_all, survtime_all = np.array([]), np.array([]), np.array([])
    probs_all, gt_all = None, np.array([])
    loss_test, grad_acc_test = 0, 0
    tp,fp,tn,fn=0,0,0,0

    for batch_idx, (x_path, x_grph, x_omic,x_mut,x_meth, censor, survtime, grade) in enumerate(test_loader):

        censor = censor.to(device) if "surv" in opt.task else censor
        grade = grade.to(device) if "grad" in opt.task else grade
        _, pred = model(x_path=x_path.to(device), x_grph=x_grph.to(device), x_omic=x_omic.to(device),x_mut=x_mut.to(device),x_meth=x_meth.to(device),training=False,dropout_rate=0)
        loss_cox = CoxLoss(survtime, censor, pred, device) if opt.task == "surv" else 0
        loss_reg = define_reg(opt, model)
        loss_nll = F.nll_loss(pred, grade) if opt.task == "grad" else 0
   #     loss_nll=lossn(pred,opt,grade) if opt.task=="grad" else 0
    #    print(loss_nll)

        loss = opt.lambda_cox*loss_cox + opt.lambda_nll*loss_nll + opt.lambda_reg*loss_reg
        loss_test += loss.data.item()
    #    print("Test loss:"+str(loss_test))
        tmp1=grade.cpu().numpy()
        gt_all = np.concatenate((gt_all,tmp1.reshape(-1)))   # Logging Information

        if opt.task == "surv":
            tmp1=pred.detach().cpu().numpy()

            risk_pred_all = np.concatenate((pred.data.item(), tmp1.reshape(-1)))   # Logging Information
            #risk_pred_all = np.concatenate((risk_pred_all, tmp1.reshape(-1)))   # Logging Information
            print(risk_pred_all)
#            print(pred)
            print(survtime)
            tmp1=censor.detach().cpu().numpy()
            censor_all = np.concatenate((censor_all, tmp1.reshape(-1)))   # Logging Information
            tmp1=survtime.detach().cpu().numpy()
            survtime_all = np.concatenate((survtime_all, tmp1.reshape(-1)))   # Logging Information
        elif opt.task == "grad":
            grade_pred = pred.argmax(dim=1, keepdim=True)
            grad_acc_test += grade_pred.eq(grade.view_as(grade_pred)).sum().item()
            tp +=(grade_pred*grade.view_as(grade_pred)).sum().item()
            fp +=(grade_pred*(grade_pred-grade.view_as(grade_pred))).sum().item()
            fn +=(grade.view_as(grade_pred)*(grade.view_as(grade_pred)-grade_pred)).sum().item()
            tn +=((1-grade_pred)*(1-grade.view_as(grade_pred))).sum().item()
            print(grade_pred.eq(grade.view_as(grade_pred)).sum().item())
            probs_np = pred.detach().cpu().numpy()
            probs_all = probs_np if probs_all is None else np.concatenate((probs_all, probs_np), axis=0)   # Logging Information
    
    ################################################### 
    # ==== Measuring Test Loss, C-Index, P-Value ==== #
    ###################################################
    #print("TEST LOADER LENGTH:")
#    print(len(test_loader))
    loss_test /=len(test_loader)
    cindex_test = CIndex_lifeline(risk_pred_all, censor_all, survtime_all) if opt.task == 'surv' else None
    pvalue_test = cox_log_rank(risk_pred_all, censor_all, survtime_all) if opt.task == 'surv' else None
    surv_acc_test = accuracy_cox(risk_pred_all, censor_all) if opt.task == 'surv' else None
    grad_acc_test = grad_acc_test / len(test_loader.dataset) if opt.task == 'grad' else None
    pred_test = [risk_pred_all, survtime_all, censor_all, probs_all, gt_all]

    return loss_test, cindex_test, pvalue_test, surv_acc_test, grad_acc_test, pred_test,tp,fp,tn,fn
