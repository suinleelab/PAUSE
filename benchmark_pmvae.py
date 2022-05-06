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

import os

def main():
    
    # get dataset, removal method
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', action="store", default='kang')
    parser.add_argument('removal', action="store", default='impute')
    parser.add_argument('which_gpu', action="store", default='0')

    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"]=args.which_gpu

    
    # load datlinger data 
    if args.dataset == 'datlinger':
        data = anndata.read('data/datlinger_pp.h5ad')
        symbols = data.var_names
    
    
     # load kang data
    if args.dataset == 'kang':
        data = anndata.read('data/kang_count.h5ad')
        symbols = data.var_names
        
        
    # load haber data
    if args.dataset == 'haber':
        data = anndata.read('/projects/leelab/data/single-cell/haber_2017/preprocessed/adata_top_2000_genes.h5ad')
        
        # filter out H poly 
        data = data[data.obs['condition'] != 'Salmonella'].copy()
        symbols = data.var_names
    
        
    # for all datasets 
    data.varm['I'] = load_annotations(
        'data/c2.cp.reactome.v7.4.symbols.gmt',
        symbols,
        min_genes=13
    ).values
    data.uns['terms'] = list(load_annotations(
        'data/c2.cp.reactome.v7.4.symbols.gmt',
        symbols,
        min_genes=13
    ).columns)
        
        
    number_of_pathways = 20
    number_of_replicates = 10
    
    logvar_results = np.zeros((number_of_replicates,number_of_pathways))
    ig_results = np.zeros((number_of_replicates,number_of_pathways))
    lr_results = np.zeros((number_of_replicates,number_of_pathways))
    kld_results = np.zeros((number_of_replicates,number_of_pathways))
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
                                            symbols,
                                            min_genes=13
                                        ).astype(bool).T
        
        ##
        ## train base model
        ##
        
        # initialize base model
        basePMVAE = pmVAEModel(membership_mask.values,
                                [12],
                                1,
                                beta=1e-05,
                                terms=membership_mask.index,
                                add_auxiliary_module=False
                            )
        
        # train
        basePMVAE.train(tr_ds, val_ds, 
                        checkpoint_path=args.dataset + '_' + args.removal +'_baseModel.pkl',
                        max_epochs=100)
        
        basePMVAE.set_gpu(False)
        
        ##
        ## get pathway rankings
        ##
        top_features = pd.DataFrame(index=data.uns['terms'])
        
        ## get max val logvar
        
        print("Calc max val score")
        
        ground_truth = torch.tensor(np.array(val_data.X)).float()
        outs = basePMVAE.model(ground_truth)
        top_features['logvar'] = -1.*outs.logvar.mean(0).detach().numpy()
        
        # IG pathway rankings
        print("Calc IG score")
        def model_loss_wrapper(z):
            module_outputs = basePMVAE.model.decoder_net(z)
            global_recon = basePMVAE.model.merge(module_outputs)
            return F.mse_loss(global_recon, ground_truth, reduction='none').mean(1).view(-1,1)
        
        input_data = outs.z
        baseline_data = torch.zeros(outs.z.shape[1])
        baseline_data.requires_grad = True
        
        explainer = PathExplainerTorch(model_loss_wrapper)
        attributions = explainer.attributions(input_data,
                                              baseline=baseline_data,
                                              num_samples=200,
                                              use_expectation=False)
        
        np_attribs = attributions.detach().numpy()
        top_features['IG'] = np_attribs.mean(0)
        
        # LR pathway rankings
        print("Calc LR score")
        
                            
        if args.dataset == 'kang' or args.dataset == 'datlinger':
            y_tr = tr_data.obs['condition']
            y_val = val_data.obs['condition']
                            
            train_labels = (y_tr == 'stimulated').values
            val_labels = (y_val == 'stimulated').values
                            
                            
        if args.dataset == 'mcfarland':
            
            y_tr = tr_data.obs['TP53_mutation_status']
            y_val = val_data.obs['TP53_mutation_status']
                            
            train_labels = (y_tr == 'Wild Type').values
            val_labels = (y_val == 'Wild Type').values
            
            
        if args.dataset == 'haber':
            y_tr = tr_data.obs['condition']
            y_val = val_data.obs['condition']
                            
            train_labels = (y_tr == 'Control').values
            val_labels = (y_val == 'Control').values
            
        if args.dataset == 'grubman': 
            y_tr = tr_data.obs['batchCond']
            y_val = val_data.obs['batchCond']
                            
            train_labels = (y_tr == 'ct').values
            val_labels = (y_val == 'ct').values
            

        if args.dataset == 'zheng': 
            y_tr = tr_data.obs['condition']
            y_val = val_data.obs['condition']
                            
            train_labels = (y_tr == 'healthy').values
            val_labels = (y_val == 'healthy').values
        
        
        
        train_embedding = basePMVAE.model(torch.tensor(tr_data.X).float()).z.detach().numpy()
        val_embedding = basePMVAE.model(torch.tensor(val_data.X).float()).z.detach().numpy()
        
        lr_scores = []
        for pathway in range(train_embedding.shape[1]):
            clf = LogisticRegression(random_state=0).fit(train_embedding[:,pathway].reshape(-1,1), train_labels)
            lr_scores.append(clf.score(val_embedding[:,pathway].reshape(-1,1), val_labels))
            
        top_features['lr_score'] = lr_scores
        top_features['lr_score'] = -1.*top_features['lr_score']
        
        # KLD pathway rankings
        print("Calc KLD")
        pathway_kld = (-0.5 * (1 + outs.logvar - outs.mu.pow(2) - outs.logvar.exp()).mean(0)).detach().numpy()
        top_features['kld'] = -1.*pathway_kld
        
        # Random pathway rankings
        print("Calc Random")
        np.random.seed(rand_seed)
        top_features['rand'] = np.random.randn(top_features.shape[0])

        # impute or retrain
        def impute_benchmark(method,n_pathways=20):
            method_recons_errors = []

            # for top 20 pathways 
            for i in range(1,1+n_pathways):

                # set pathways = 0.
                test_matrix = torch.tensor(test_data.X).float()
                test_matrix_embedded = basePMVAE.model(test_matrix).z
                for x in top_features.sort_values(method).index[:i]:
                    index_to_zero = list(top_features.index).index(x)
                    test_matrix_embedded[:,index_to_zero] = 0.

                module_outputs = basePMVAE.model.decoder_net(test_matrix_embedded)
                global_recon = basePMVAE.model.merge(module_outputs)
                recons_error = F.mse_loss(global_recon, test_matrix).detach().item()
                method_recons_errors.append(recons_error)
            return method_recons_errors
        
        def retrain_benchmark(method,n_pathways=20):
            method_recons_errors = []
            # for top 20 pathways 
            for i in range(1,21):

                # get cumulative pathways
                A_new=[]
                for x in top_features.sort_values(method).index[:i]:
                    A_new.append(membership_mask.loc[x,:].values.reshape(1,-1))
                A_new = np.concatenate(A_new,axis=0)

                reducedVAE = pmVAEModel(
                                A_new,
                                [12],
                                1,
                                beta=1e-05,
                                terms=list(range(A_new.shape[0])),
                                add_auxiliary_module=False
                            )
                
                reducedVAE.train(tr_ds, val_ds, checkpoint_path= args.dataset + '_' + args.removal +'_reducedVAE.pkl', max_epochs=50)

                test_matrix = torch.tensor(test_data.X).float().cuda()
                global_recon = reducedVAE.model(test_matrix).global_recon

                recons_error = F.mse_loss(global_recon, test_matrix).detach().item()
                method_recons_errors.append(recons_error)
            return method_recons_errors
                
                 
        # run impute or retrain 
        if args.removal == "impute": 
            print("Impute Logvar")
            logvar_results[rand_seed,:] = impute_benchmark('logvar')
            print("Impute IG")
            ig_results[rand_seed,:] = impute_benchmark('IG')
            print("Impute LR")
            lr_results[rand_seed,:] = impute_benchmark('lr_score')
            print("Impute KLD")
            kld_results[rand_seed,:] = impute_benchmark('kld')
            print("Impute RAND")
            rand_results[rand_seed,:] = impute_benchmark('rand')
            
        if args.removal == "retrain":
            print("Retrain Logvar")
            logvar_results[rand_seed,:] = retrain_benchmark('logvar')
            print("Retrain IG")
            ig_results[rand_seed,:] = retrain_benchmark('IG')
            print("Retrain LR")
            lr_results[rand_seed,:] = retrain_benchmark('lr_score')
            print("Retrain KLD")
            kld_results[rand_seed,:] = retrain_benchmark('kld')
            print("Retrain RAND")
            rand_results[rand_seed,:] = retrain_benchmark('rand')
                      
                    
        # save results every iteration so that if it crashes
        # there's at least some progress
        with open('results/{}_{}_logvar.npy'.format(args.dataset, args.removal), 'wb') as f:
            np.save(f, logvar_results)
        with open('results/{}_{}_ig.npy'.format(args.dataset, args.removal), 'wb') as f:
            np.save(f, ig_results)
        with open('results/{}_{}_lr.npy'.format(args.dataset, args.removal), 'wb') as f:
            np.save(f, lr_results)
        with open('results/{}_{}_kld.npy'.format(args.dataset, args.removal), 'wb') as f:
            np.save(f, kld_results)
        with open('results/{}_{}_rand.npy'.format(args.dataset, args.removal), 'wb') as f:
            np.save(f, rand_results)
    
    
if __name__ == '__main__':
    main()    
