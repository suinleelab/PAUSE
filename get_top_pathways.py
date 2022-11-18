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
import argparse


from models import pmVAEModel
import mygene
import os
import time 

save_path = 'new_for_revision/new_res/'

def main():

    ig_times = []
    lr_times = []
    train_times = []
    
    # get dataset, removal method
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', action="store", default='kang')
    parser.add_argument('which_gpu', action="store", default='0')
    parser.add_argument('gene_prog', action="store", default='Ctrl')

    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"]=args.which_gpu
    dataset =args.dataset


    # load data 
    
    # load datlinger data 
    if args.dataset == 'datlinger':
        
        data = anndata.read('data/datlinger_pp.h5ad')
        symbols = data.var_names
    
    
     # load kang data
    if args.dataset == 'kang':
        
        data = anndata.read('data/kang_count.h5ad')
        symbols = data.var_names
        
    
    # load mcfarland data
    if args.dataset == 'mcfarland':
        
        data = anndata.read('/projects/leelab/data/single-cell/mcfarland_2020_Idasanutlin/preprocessed/adata_top_2000_genes_tc.h5ad')
        data = data[data.obs['condition'] == 'Idasanutlin'].copy() 
        symbols = data.var_names
                   
            
    # load zheng data 
    if args.dataset == 'zheng':
        data = anndata.read('/projects/leelab/data/single-cell/zheng_2017/preprocessed/adata_top_2000_genes.h5ad')

        # convert ENSG IDs to gene symbols: 
        
        mg = mygene.MyGeneInfo()
        geneList = data.var_names
        geneSyms = mg.querymany(geneList , scopes='ensembl.gene', fields='symbol', species='human', returnall=True)

        symbols = []
        not_in = []
        is_in = []
        for k in range(2000):
            if ('symbol' in geneSyms['out'][k]):  
                symbols += [geneSyms['out'][k]['symbol']]
                is_in += [geneSyms['out'][k]['query']]
            else:
                not_in += [geneSyms['out'][k]['query']]
        symbols = pd.Index(symbols)
        
        symbols = pd.Index(set(symbols.to_numpy()))

        # filter out post transplant
        data = data[data.obs['condition'] != 'post_transplant'][:,is_in].copy() 
        
            
    # load haber data
    if args.dataset == 'haber':
        
        data = anndata.read('/projects/leelab/data/single-cell/haber_2017/preprocessed/adata_top_2000_genes.h5ad')
        
        # filter out H poly 
        data = data[data.obs['condition'] != 'Salmonella'].copy()
       
        symbols = data.var_names
    

    # load grubman data 
    if args.dataset == 'grubman':
        
        data = anndata.read('/projects/leelab/data/single-cell/grubman_2019/preprocessed/adata_top_2000_genes.h5ad')
       
        symbols = data.var_names  
    
    
    if args.dataset == 'norman': 
                
        data = anndata.read('/projects/leelab/data/single-cell/norman_2019/preprocessed/adata_top_2000_genes_tc.h5ad')
        
        if args.gene_prog == 'erythroid': 
            data = data[(data.obs['gene_program'] == 'Ctrl') | (data.obs['gene_program'] == 'Erythroid')].copy()
                        
        if args.gene_prog == 'granulocyte-apoptosis': 
            data = data[(data.obs['gene_program'] == 'Ctrl') | (data.obs['gene_program'] == 'Granulocyte/apoptosis')].copy()

        if args.gene_prog == 'megakaryocyte': 
            data = data[(data.obs['gene_program'] == 'Ctrl') | (data.obs['gene_program'] == 'Megakaryocyte')].copy()
            
        if args.gene_prog == 'pro-growth': 
            data = data[(data.obs['gene_program'] == 'Ctrl') | (data.obs['gene_program'] == 'Pro-growth')].copy()
            
        test_df = pd.DataFrame(index=data.var['gene_name'])
        symbols = test_df.index
        

        
    number_of_replicates = 10
    first_run = True 
    
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
        ## train model
        ##

        # initialize base model
        basePMVAE = pmVAEModel(membership_mask.values,
                                [12],
                                1,
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
                        checkpoint_path='saved_models/' + dataset + '_' + args.gene_prog + '.pkl',
                        max_epochs=100)
        
        end_train = time.time()
        train_times.append(end_train - start_train)

        basePMVAE.set_gpu(False)


        # IG pathway rankings
        print("Calc IG score")
        
        start_ig = time.time()
        
        def model_loss_wrapper(z):
            module_outputs = basePMVAE.model.decoder_net(z)
            global_recon = basePMVAE.model.merge(module_outputs)
            return F.mse_loss(global_recon, ground_truth, reduction='none').mean(1).view(-1,1)

        ground_truth = torch.tensor(np.array(val_data.X)).float()
        outs = basePMVAE.model(ground_truth)
        
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
        top_ig.to_csv(save_path + dataset + '_ig.csv', index=False)
        
        
        # LR pathway rankings
        print("Calc LR score")
        start_lr = time.time()

        
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
            
        
        if args.dataset == 'norman': 
            y_tr = tr_data.obs['gene_program']
            y_val = val_data.obs['gene_program']

            train_labels = (y_tr == 'Ctrl').values
            val_labels = (y_val == 'Ctrl').values
        
        train_embedding = basePMVAE.model(torch.tensor(tr_data.X).float()).z.detach().numpy()
        val_embedding = basePMVAE.model(torch.tensor(val_data.X).float()).z.detach().numpy()

        lr_scores = []
        for pathway in range(train_embedding.shape[1]):
            clf = LogisticRegression(random_state=0).fit(train_embedding[:,pathway].reshape(-1,1), train_labels)
            lr_scores.append(clf.score(val_embedding[:,pathway].reshape(-1,1), val_labels))
            
        
        top_lr[rand_seed] = lr_scores
        top_lr[rand_seed] = -1.*top_lr[rand_seed]
        
        end_lr = time.time()
        lr_times.append(end_lr - start_lr)


        # so far! 
        top_lr.to_csv(save_path + dataset + '_lr.csv', index=False)

        times = pd.DataFrame()
        times['ig_times'] = ig_times
        times['lr_times'] = lr_times
        times['train_times'] = train_times

        times.to_csv(save_path + args.dataset + '_times.csv')

if __name__ == '__main__':
    main()    