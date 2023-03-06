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
## load a single cell dataset
data = anndata.read('data/datlinger_pp.h5ad')

## load a pathway file 
## more examples can be found here (http://www.gsea-msigdb.org/gsea/msigdb/collections.jsp)
data.varm['annotations'] = load_annotations(
    'data/c2.cp.reactome.v7.4.symbols.gmt',
    data.var_names,
    min_genes=13
)
```

## Reproducing experiments and figures from paper

For code to generate the models used, see "models.py". Pathway attributions and gene attributions are generated using code from "pathexplainer.py". Benchmarking pathways attributions against other methods for ranking pathway importance is done using the files "benchmark_pmvae.py", "benchmark_intercode.py", and "top_pathways.py". For code to generate the figures in the paper, see the folder `figures`. 

