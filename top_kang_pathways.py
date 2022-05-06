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

from models import pmVAEModel
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"



def main():

    # load data 
    data = anndata.read('data/kang_count.h5ad')
    symbols = data.var_names
    
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
        ## train base model
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
        basePMVAE.train(tr_ds, val_ds, 
                        checkpoint_path='top_kang.pkl',
                        max_epochs=100)

        basePMVAE.set_gpu(False)


        # IG pathway rankings
        print("Calc IG score")
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

        # so far! 
        top_ig.to_csv('aux_one_kang_ig.csv', index=False)
        
        
        
        
        # LR pathway rankings
        print("Calc LR score")
        
        y_tr = tr_data.obs['condition']
        y_val = val_data.obs['condition']

        train_labels = (y_tr == 'stimulated').values
        val_labels = (y_val == 'stimulated').values
        
        
        train_embedding = basePMVAE.model(torch.tensor(tr_data.X).float()).z.detach().numpy()
        val_embedding = basePMVAE.model(torch.tensor(val_data.X).float()).z.detach().numpy()

        lr_scores = []
        for pathway in range(train_embedding.shape[1]):
            clf = LogisticRegression(random_state=0).fit(train_embedding[:,pathway].reshape(-1,1), train_labels)
            lr_scores.append(clf.score(val_embedding[:,pathway].reshape(-1,1), val_labels))
            
        
        top_lr[rand_seed] = lr_scores
        top_lr[rand_seed] = -1.*top_lr[rand_seed]


        # so far! 
        top_lr.to_csv('aux_one_kang_lr.csv', index=False)
        
        
    
    top_ig.to_csv('kang_ig.csv', index=False)
    top_lr.to_csv('kang_lr.csv', index=False)

    
    
if __name__ == '__main__':
    main()    