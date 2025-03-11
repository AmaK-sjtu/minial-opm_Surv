import os
import pickle
from lifelines.utils import concordance_index
from lifelines import CoxPHFitter
import numpy as np
import pandas as pd
pd.options.display.max_rows = 999
from utils import CI_pm
from utils import cox_log_rank
from utils import getCleanAllDataset, addHistomolecularSubtype
from utils import makeKaplanMeierPlot

def trainCox(dataroot = './data/BRCA/', ckpt_name='./checkpoints/BRCA/surv10/', model='omic', penalizer=1e-4):
    if not os.path.exists(ckpt_name): os.makedirs(ckpt_name)
    if not os.path.exists(os.path.join(ckpt_name, model)): os.makedirs(os.path.join(ckpt_name, model))
    pnas_splits = pd.read_csv(dataroot+'pnas_splits.csv')
    pnas_splits.columns = ['ID']+[str(k) for k in range(1, 11)]
    pnas_splits.index = pnas_splits['ID']
    pnas_splits = pnas_splits.drop(['ID'], axis=1)
    ignore_missing_moltype=True
    ignore_missing_histype=True
    all_dataset = getCleanAllDataset(dataroot=dataroot, ignore_missing_moltype=ignore_missing_moltype, 
                                     ignore_missing_histype=ignore_missing_histype)[1]
    model_feats = {'cox_omic':['ID', 'Histology', 'Grade', 'Molecular subtype'],
                   'cox_histype':['Survival months', 'censored', 'Histology'],
                   'cox_grade':['Survival months', 'censored', 'Grade']}
    cv_results = []
    for k in pnas_splits.columns:
        if k==0:
            pat_train = list(set(pnas_splits.index[pnas_splits[k] == 'Train']).intersection(all_dataset.index))
            pat_test = list(set(pnas_splits.index[pnas_splits[k] == 'Test']).intersection(all_dataset.index))
            feats = all_dataset.columns.drop(model_feats[model]) if model == 'omic' or model == 'all' else model_feats[model]
            train = all_dataset.loc[pat_train]
            test = all_dataset.loc[pat_test]

            cph = CoxPHFitter(penalizer=penalizer)
            cph.fit(train[feats], duration_col='Survival months', event_col='censored', show_progress=False)
            cin = concordance_index(test['Survival months'], -cph.predict_partial_hazard(test[feats]), test['censored'])
            cv_results.append(cin)
            
            train.insert(loc=0, column='Hazard', value=-cph.predict_partial_hazard(train))
            test.insert(loc=0, column='Hazard', value=-cph.predict_partial_hazard(test))
            pickle.dump(train, open(os.path.join(ckpt_name, model, '%s_%s_pred_train.pkl' % (model, k)), 'wb'))
            pickle.dump(test, open(os.path.join(ckpt_name, model, '%s_%s_pred_test.pkl' % (model, k)), 'wb'))
        
    pickle.dump(cv_results, open(os.path.join(ckpt_name, model, '%s_results.pkl' % model), 'wb'))
    print("C-Indices across Splits", cv_results)
    print("Average C-Index: %f" % CI_pm(cv_results))

for model in ['omic', 'graph', 'graphomic']:
    makeKaplanMeierPlot(ckpt_name='./checkpoints/NSCLC/surv10/', model=model, split='test')
