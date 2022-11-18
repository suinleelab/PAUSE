# get Mcfarland top pathways, condition on cell lines 

import anndata
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import os

from utils import load_annotations
from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset, DataLoader
from datasets import RNASeqData

from pathexplainer import PathExplainerTorch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
import argparse


from models import pmVAEModel
import os
import time 

save_path = 'new_for_revision/new_res/'


def main():

    ig_times = []
    lr_times = []
    train_times = []
    
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', action="store", default='kang')
    parser.add_argument('which_gpu', action="store", default='0')

    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"]=args.which_gpu
    dataset =args.dataset

    # load data 
    
    # load mcfarland data
    data = anndata.read('/projects/leelab/data/single-cell/mcfarland_2020_Idasanutlin/preprocessed/adata_top_2000_genes_tc.h5ad')
    
    data = data[data.obs['condition'] == 'Idasanutlin'].copy() 
    symbols = data.var_names
    
    conditions = np.array(data.obs['cell_line']).reshape(-1,1)
    enc = OneHotEncoder()
    enc.fit(conditions)
    pre_processed_conditions = enc.transform(conditions).toarray()
    
    number_of_replicates = 10

    first_run = True 
    
    # for 10 experimental replicates
    for rand_seed in range(number_of_replicates):

        print("replicate number " + str(rand_seed))

        # split data
        
        train_data, test_data, train_c, test_c = train_test_split(data,pre_processed_conditions,
                                                test_size=0.25,
                                                shuffle=True,
                                                random_state=rand_seed)
        tr_data, val_data, tr_c, val_c = train_test_split(train_data,train_c,
                                            test_size=0.25,
                                            shuffle=True,
                                            random_state=rand_seed)
        
        tr_ds = RNASeqData(np.array(tr_data.X), c=tr_c)
        val_ds = RNASeqData(np.array(val_data.X), c=val_c)

        # load annotations
        membership_mask = load_annotations('data/c2.cp.reactome.v7.4.symbols.gmt',
                                            symbols,
                                            min_genes=13
                                        ).astype(bool).T

        ##
        ## train model
        ##

        # initialize base model
        basePMVAE = pmVAEModel(membership_mask.values,
                                [12],
                                1,
                                cdim = tr_c.shape[1],
                                beta=1e-05,
                                terms=membership_mask.index,
                                add_auxiliary_module=True
                            )

        
        if first_run: # first run 
            top_ig = pd.DataFrame(index=basePMVAE.latent_space_names())
            top_lr = pd.DataFrame(index=basePMVAE.latent_space_names())
            first_run = False 
        
        
        # train
        
        start_train = time.time()
        basePMVAE.train(tr_ds, val_ds, 
                        checkpoint_path='saved_models/seed_' + str(rand_seed) + 'cell_lines_cond_top_' + dataset + '.pkl',
                        max_epochs=100)
        
        end_train = time.time()
        train_times.append(end_train - start_train)
        

        basePMVAE.set_gpu(False)


        # IG pathway rankings
        print("Calc IG score")
        
        start_ig = time.time()
        
        def model_loss_wrapper(z):
            latent_input = torch.cat([z, c_full], 1)
            module_outputs = basePMVAE.model.decoder_net(latent_input)
            global_recon = basePMVAE.model.merge(module_outputs)
            return F.mse_loss(global_recon, ground_truth, reduction='none').mean(1).view(-1,1)

        ground_truth = torch.tensor(data.X).float()
        c_full = torch.tensor(pre_processed_conditions).float()
        outs = basePMVAE.model(ground_truth,c_full)
        
        input_data = outs.z
        baseline_data = torch.zeros(outs.z.shape[1])
        baseline_data.requires_grad = True

        explainer = PathExplainerTorch(model_loss_wrapper)
        attributions = explainer.attributions(input_data,
                                      baseline=baseline_data,
                                      num_samples=200,
                                      use_expectation=False)

        np_attribs = attributions.detach().numpy()
        top_ig[rand_seed] = np_attribs.mean(0)
        
        end_ig = time.time()
        ig_times.append(end_ig - start_ig)
        

        # so far! 
        top_ig.to_csv(save_path + dataset + '_cell_lines_cond_ig.csv', index=False)
        
        
        # LR pathway rankings
        print("Calc LR score")
        start_lr = time.time()

        if args.dataset == 'mcfarland':

            y_tr = tr_data.obs['TP53_mutation_status']
            y_val = val_data.obs['TP53_mutation_status']

            train_labels = (y_tr == 'Wild Type').values
            val_labels = (y_val == 'Wild Type').values
            
        
        train_embedding = basePMVAE.model(torch.tensor(tr_data.X).float(), torch.tensor(tr_c).float()).z.detach().numpy()
        val_embedding = basePMVAE.model(torch.tensor(val_data.X).float(), torch.tensor(val_c).float()).z.detach().numpy()
        
        
        lr_scores = []
        for pathway in range(train_embedding.shape[1]):
            clf = LogisticRegression(random_state=0).fit(train_embedding[:,pathway].reshape(-1,1), train_labels)
            lr_scores.append(clf.score(val_embedding[:,pathway].reshape(-1,1), val_labels))
            
        
        top_lr[rand_seed] = lr_scores
        top_lr[rand_seed] = -1.*top_lr[rand_seed]
        
        end_lr = time.time()
        lr_times.append(end_lr - start_lr)


        # so far! 
        top_lr.to_csv(save_path + dataset + '_cell_lines_cond_lr.csv', index=False)

        times = pd.DataFrame()
        times['ig_times'] = ig_times
        times['lr_times'] = lr_times
        times['train_times'] = train_times

        times.to_csv(save_path + args.dataset + '_cell_lines_cond_times.csv')
    
    
if __name__ == '__main__':
    main()    