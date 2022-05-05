#!/usr/bin/env python
# benchmark_pmvae.py

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

import argparse

from pathexplainer import PathExplainerTorch
from sklearn.linear_model import LogisticRegression

from models import pmVAEModel

from intercode import AutoencoderLinearDecoder, train_autoencoder

import os
os.environ["CUDA_VISIBLE_DEVICES"]="4"

def main():
    
    # get dataset, removal method
    parser = argparse.ArgumentParser()
#     parser.add_argument('split', action="store", default='0')
    parser.add_argument('dataset', action="store", default='kang')
    parser.add_argument('removal', action="store", default='impute')
    
    args = parser.parse_args()
    
    # load data
    if args.dataset == 'kang':
        
        data = anndata.read('data/kang_count.h5ad')
        data.varm['I'] = load_annotations(
            'data/c2.cp.reactome.v7.4.symbols.gmt',
            data.var_names,
            min_genes=13
        ).values
        data.uns['terms'] = list(load_annotations(
            'data/c2.cp.reactome.v7.4.symbols.gmt',
            data.var_names,
            min_genes=13
        ).columns)
        
    number_of_pathways = 20
    number_of_replicates = 10
    
    l2_results = np.zeros((number_of_replicates,number_of_pathways))
    ig_results = np.zeros((number_of_replicates,number_of_pathways))
#     lr_results = np.zeros((number_of_replicates,number_of_pathways))
#     kld_results = np.zeros((number_of_replicates,number_of_pathways))
    rand_results = np.zeros((number_of_replicates,number_of_pathways))
    
    # for 10 experimental replicates
    for rand_seed in range(number_of_replicates):
        
        print("replicate number " + str(rand_seed))
        
        # split data
        
        train_data, test_data = train_test_split(data,
                                                test_size=0.25,
                                                shuffle=True,
                                                random_state=rand_seed)
        tr_data, val_data = train_test_split(train_data,
                                            test_size=0.25,
                                            shuffle=True,
                                            random_state=rand_seed)
        
        tr_ds = RNASeqData(np.array(tr_data.X))
        val_ds = RNASeqData(np.array(val_data.X))
        
        # load annotations
        membership_mask = load_annotations('data/c2.cp.reactome.v7.4.symbols.gmt',
                                            data.var_names,
                                            min_genes=13
                                        ).astype(bool).T
        
        ##
        ## train base model
        ##
        
        LR = 0.001
        BATCH_SIZE = 62
        N_EPOCHS = 30

        # regularization hyperparameters
        # lambda0 - page 19 of presentation
        # lambdas 1-3 - last term on page 20

        LAMBDA0 = 0.1

        LAMBDA1 = 0.93*LR
        LAMBDA2 = 0.43*LR
        LAMBDA3 = 0.57*LR
        
        # initialize base model
        autoencoder = AutoencoderLinearDecoder(tr_data.n_vars, n_ann=len(tr_data.uns['terms']))
        autoencoder.cuda()
        
        # train
        train_autoencoder(tr_data, autoencoder, LR, BATCH_SIZE, N_EPOCHS,
                  l2_reg_lambda0=LAMBDA0, lambda1=LAMBDA1, lambda2=LAMBDA2, lambda3=LAMBDA3)
        
        ##
        ## get pathway rankings
        ##
        top_features = pd.DataFrame(index=data.uns['terms'])
        
        ## get L2
        top_features['l2'] = -1.*autoencoder.decoder.weight_dict['annotated'].data.norm(p=2, dim=0).detach().cpu().numpy()
        
        print("Calc IG score")
        # IG pathway rankings
        ground_truth = torch.tensor(val_data.X).float()
        autoencoder.cpu()

        def intercode_loss_wrapper(z):
            global_recon = autoencoder.decoder(z)
            return F.mse_loss(global_recon, ground_truth, reduction='none').mean(1).view(-1,1)

        
        input_data = autoencoder.encoder(torch.tensor(val_data.X).float())
        baseline_data = torch.zeros(input_data.shape[1])
        baseline_data.requires_grad = True
        
        explainer = PathExplainerTorch(intercode_loss_wrapper)
        attributions = explainer.attributions(input_data,
                                              baseline=baseline_data,
                                              num_samples=200,
                                              use_expectation=False)
        
        top_features['IG'] = attributions.detach().numpy().mean(0)
        
#         # LR pathway rankings
#         print("Calc LR score")
#         y_tr = tr_data.obs['condition']
#         y_val = val_data.obs['condition']
        
#         train_embedding = basePMVAE.model(torch.tensor(tr_data.X).float()).z.detach().numpy()
#         val_embedding = basePMVAE.model(torch.tensor(val_data.X).float()).z.detach().numpy()
        
#         lr_scores = []
#         for pathway in range(train_embedding.shape[1]):
#             train_labels = (y_tr == 'stimulated').values
#             val_labels = (y_val == 'stimulated').values
#             clf = LogisticRegression(random_state=0).fit(train_embedding[:,pathway].reshape(-1,1), train_labels)
#             lr_scores.append(clf.score(val_embedding[:,pathway].reshape(-1,1), val_labels))
            
#         top_features['lr_score'] = lr_scores
#         top_features['lr_score'] = -1.*top_features['lr_score']
        
#         # KLD pathway rankings
#         print("Calc KLD")
#         pathway_kld = (-0.5 * (1 + outs.logvar - outs.mu.pow(2) - outs.logvar.exp()).mean(0)).detach().numpy()
#         top_features['kld'] = -1.*pathway_kld
        
        # Random pathway rankings
        print("Calc Random")
        np.random.seed(rand_seed)
        top_features['rand'] = np.random.randn(top_features.shape[0])

        # impute or retrain
        def impute_benchmark(method,n_pathways=20):
            method_recons_errors = []

            # for top 10 pathways 
            for i in range(1,1+n_pathways):

                # set pathways = 0.
                test_matrix = torch.tensor(test_data.X).float()
                test_matrix_embedded = autoencoder.encoder(test_matrix)
                for x in top_features.sort_values(method).index[:i]:
                    index_to_zero = list(top_features.index).index(x)
                    test_matrix_embedded[:,index_to_zero] = 0.

                global_recon = autoencoder.decoder(test_matrix_embedded)
                recons_error = F.mse_loss(global_recon, test_matrix).detach().item()
                method_recons_errors.append(recons_error)
            return method_recons_errors
        
        print("Impute L2")
        l2_results[rand_seed,:] = impute_benchmark('l2')
        print("Impute IG")
        ig_results[rand_seed,:] = impute_benchmark('IG')
#         print("Impute LR")
#         lr_results[rand_seed,:] = impute_benchmark('lr_score')
#         print("Impute KLD")
#         kld_results[rand_seed,:] = impute_benchmark('kld')
        print("Impute RAND")
        rand_results[rand_seed,:] = impute_benchmark('rand')

    # save results
    with open('results/intercode_kang_impute_l2.npy', 'wb') as f:
        np.save(f, l2_results)
    with open('results/intercode_kang_impute_ig.npy', 'wb') as f:
        np.save(f, ig_results)
#     with open('results/intercode_kang_impute_lr.npy', 'wb') as f:
#         np.save(f, lr_results)
#     with open('results/intercode_kang_impute_kld.npy', 'wb') as f:
#         np.save(f, kld_results)
    with open('results/intercode_kang_impute_rand.npy', 'wb') as f:
        np.save(f, rand_results)
    
if __name__ == '__main__':
    main()    