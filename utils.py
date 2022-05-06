import pandas as pd

def parse_gmt(path, symbols=None, min_genes=10):
    lut = dict()
    for line in open(path, 'r'):
        key, _, *genes = line.strip().split()
        if symbols is not None:
            genes = symbols.intersection(genes).tolist()
        if len(genes) < min_genes:
            continue
        lut[key] = genes

    return lut

def load_annotations(gmt, genes, min_genes=10):
    genesets = parse_gmt(gmt, genes, min_genes)
    annotations = pd.DataFrame(False, index=genes, columns=genesets.keys())
    for key, genes in genesets.items():
        annotations.loc[genes, key] = True

    return annotations