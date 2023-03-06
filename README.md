# PAUSE

<center>
    <img src="./concept_fig.png?raw=true" width="750">
</center>

Code for the paper "Principled feature attribution for unsupervised gene expression analysis" (PAUSE). 
For more information, see our preprint: https://www.biorxiv.org/content/10.1101/2022.05.03.490535v1.

## Examples

### Identify most important pathways from an interpretable autoencoder
This first example demonstrates how the PAUSE framework can be used to identify the most important pathways for an interpretable autoencoder.

```python
import anndata
# and other import statements...

## load a single cell dataset
data = anndata.read('data/datlinger_pp.h5ad')

## load a pathway gene set file 
## more examples can be found here (http://www.gsea-msigdb.org/gsea/msigdb/collections.jsp)
data.varm['annotations'] = load_annotations(
    'data/c2.cp.reactome.v7.4.symbols.gmt',
    data.var_names,
    min_genes=13
)
# binary matrix mapping from genes to pathways
membership_mask = data.varm['annotations'].astype(bool).T.values
```

After loading the RNA-seq dataset you want to analyze, you can then initialize and train a model on the dataset. In this case, we use our PyTorch implementation of the [pmVAE architecture](https://www.biorxiv.org/content/10.1101/2021.01.28.428664v1), which is a variational autoencoder composed of a set of subnetworks (pathway modules) that are factorized according to the gene sets defined above. In this model, each latent node in the bottleneck layer only contains information about the genes belonging to its corresponding pathway.

```python
from models import pmVAEModel 

# initialize pmVAE model. 
# positional arguments are 1) the binary gene set membership matrix, 
# 2) a list containing the number of nodes in each hidden layer, and 
# 3) an integer indicating the number of nodes in each module's bottleneck.
pmvae = pmVAEModel(
    membership_mask,
    [6], # This indicates that there will be one intermediate layer before the bottleneck with 6 nodes in each module. To have 2 intermediate layers of 6 nodes, you could write [6, 6]
    4, # number of nodes in each module bottleneck 
    terms=membership_mask.index, # a list of the names of the pathway modules
    add_auxiliary_module=True # whether or not to include a densely connected auxiliary module
)

# train pmVAE model
pmvae.train(train_dataset, # a PyTorch dataset object containing the training expression samples
              val_dataset, # a PyTorch dataset object containing the val expression samples
              max_epochs=200, # Maximum number of epochs to train
              lr=0.001, # learning rate of the adam optimizer used to train the model
              beta=1e-5, # weight multiplier of KL loss term
              batch_size=256, # samples per batch
              pathway_dropout=True, # whether or not to train with pathway dropout scheme as defined in pmVAE paper
              checkpoint_path='pmvae_checkpoint.pkl' # path of model checkpoint
              )
```
## Reproducing experiments and figures from paper

For code to generate the models used, see "models.py". Pathway attributions and gene attributions are generated using code from "pathexplainer.py". Benchmarking pathways attributions against other methods for ranking pathway importance is done using the files "benchmark_pmvae.py", "benchmark_intercode.py", and "top_pathways.py". For code to generate the figures in the paper, see the folder `figures`. 

