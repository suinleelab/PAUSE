# PAUSE

![Main Concept Fig](/images/conceptFig.jpg)

Code for the paper "Principled feature attribution for unsupervised gene expression analysis" (PAUSE). 
For more information, see our preprint: https://www.biorxiv.org/content/10.1101/2022.05.03.490535v1.

## Examples

### Identify most important pathways from an interpretable autoencoder
This first example demonstrates how the PAUSE framework can be used to identify the most important pathways for an interpretable autoencoder.

```python
import anndata
# and other import statements...

## load a single cell dataset
data = anndata.read('data/kang_count.h5ad')

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
    [12], # This indicates that there will be one intermediate layer before the bottleneck with 12 nodes in each module. To have 2 intermediate layers of 6 nodes, you could write [6, 6]
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

Once the model is trained, we can use the [Path Explain software](https://github.com/suinleelab/path_explain) (also provided in this repository in the `pathexplainer.py` file) to *identify the top pathways* in the dataset by explaining the trained models reconstruction error with respect to the learned latent pathways.

```python
from pathexplainer import PathExplainerTorch
import torch
import torch.nn.functional as F

# define a wrapper function that outputs the reconstruction error of the model given the latent codes
def model_loss_wrapper(z):
    module_outputs = pmvae.model.decoder_net(z)
    global_recon = pmvae.model.merge(module_outputs)
    return F.mse_loss(global_recon, ground_truth, reduction='none').mean(1).view(-1,1)
    
# define a tensor to hold the original data, which gets used as an argument in the reconstruction error in the wrapper above
ground_truth = torch.tensor(data.X).float()

# get the latent codes to use as input to the model loss wrapper
outs = pmvae.model(ground_truth)
input_data = outs.z
baseline_data = torch.zeros(outs.z.shape[1]) # define a baseline, in this case the zeros vector
baseline_data.requires_grad = True

# calculate the pathway attributions
explainer = PathExplainerTorch(model_loss_wrapper)
attributions = explainer.attributions(input_data,
                                      baseline=baseline_data,
                                      num_samples=200, # number of samples to use when calculating the path integral
                                      use_expectation=False)

```

Once you have calculated the pathway attributions, you can average them over all samples in the dataset to identify and plot the most important pathways.

```python
# move attributions to numpy, make a df w/ index as latent space names
np_attribs = attributions.detach().numpy()
top_features = pd.DataFrame(index=pmvae.latent_space_names())
top_features['global_attribs'] = np_attribs.mean(0) # in this case, global attributions are the mean over the dataset

# Loss explanation
top_features.sort_values('global_attribs',ascending=True).iloc[:30,0].plot.bar()
```

![Showing pathway attributions](/images/top_pathways_img.png)

### Identify most important genes contributing to a particular latent pathway
This first example demonstrates how the PAUSE framework can be used to identify the most important pathways for an interpretable autoencoder. However, as you see above, for the dataset in question, we can see that the most important pathways are the "uninterpretable" densely-connected auxiliary pathways. How can we identify the most important genes contributing to these latent pathways, and interpret their biological meaning? By using gene level attributions. This example uses the same trained pmVAE model as the above example. We can now take that model

```python

# explain tcr in terms of genes
def model_latent_wrapper(x):
    outs = pmvae.model(x)
    z = outs.mu
    return z[:,316].reshape(-1,1)

```

## Reproducing experiments and figures from paper

For code to generate the models used, see "models.py". Pathway attributions and gene attributions are generated using code from "pathexplainer.py". Benchmarking pathways attributions against other methods for ranking pathway importance is done using the files "benchmark_pmvae.py", "benchmark_intercode.py", and "top_pathways.py". For code to generate the figures in the paper, see the folder `figures`. 

