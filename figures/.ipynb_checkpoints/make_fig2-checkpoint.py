import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sb
from statannotations.Annotator import Annotator

DATASETS = ['kang', 'haber', 'datlinger']

METHODS = ['impute', 'retrain'] 
PAL = 'colorblind'

LABEL_SIZE = 18
TITLE_SIZE = 18
AXES_SIZE = 18
LEG_SIZE = 14


def get_arrays(dataset, method): 
    ig_results = np.load('complete_results/'+ dataset + '_' + method + '_ig.npy')
    logvar_results = np.load('complete_results/'+ dataset + '_' + method + '_logvar.npy')
    lr_results = np.load('complete_results/'+ dataset + '_' + method + '_lr.npy')
    kld_results = np.load('complete_results/'+ dataset + '_' + method + '_kld.npy')
    rand_results = np.load('complete_results/'+ dataset + '_' + method + '_rand.npy')

    return ig_results, logvar_results, lr_results, kld_results, rand_results 

# load results for single dataset and benchmark method 
def load_res(dataset, method): 
    
    ig_results, logvar_results, lr_results, kld_results, rand_results = get_arrays(dataset, method)
    
    # get AUCs
    ig_aucs = np.trapz(ig_results, axis=1)
    lr_aucs = np.trapz(lr_results, axis=1)
    kld_aucs = np.trapz(kld_results, axis=1)
    rand_aucs = np.trapz(rand_results, axis=1)
    logvar_aucs = np.trapz(logvar_results, axis=1)
    
    auc_stack = np.concatenate((ig_aucs, lr_aucs, kld_aucs, rand_aucs, logvar_aucs))
    
    num_trials = 10 
    #rankings_methods = np.concatenate((['Loss Attribution']*num_trials, ['LR Score']*num_trials, ['KL Divergence']*num_trials, ['Random']*num_trials, ['LS Variance']*num_trials))
    
    rankings_methods = np.concatenate((['PAUSE']*num_trials, ['LR']*num_trials, ['KLD']*num_trials, ['Random']*num_trials, ['LSV']*num_trials))

    
    
    results = pd.DataFrame(index=list(range(0,50)))
    results['methods'] = rankings_methods
    results['aucs'] = auc_stack
    
    return results


def get_subplot(dataset, method): 

    plt.rc('axes', titlesize=TITLE_SIZE)     # fontsize title
    plt.rc('axes', labelsize=AXES_SIZE)    # fontsize of the x and y axis labels
    plt.rc('xtick', labelsize=LABEL_SIZE)    # fontsize of the (method) tick labels
    
    results = load_res(dataset, method) 
    
    plt.style.use('seaborn-colorblind')

    fig, ax = plt.subplots(figsize=(6,4))

    bp = sb.boxplot(ax=ax,
             data=results,x='methods',y='aucs',dodge=True,
             color='white', fliersize=0, 
           )

    sb.stripplot(ax=ax,
             data=results,x='methods',y='aucs',
             dodge=True,
             s=4)
    
    """
    pairs=[("Loss Attribution", "LR Score")]
    annotator = Annotator(ax, pairs, data=results, x='methods',y='aucs')
    annotator.set_custom_annotations(['**'])
    annotator.annotate()
    """
    
    # for ** position
    top = [results[results['methods'] == "LR"].max()['aucs'], 
            results[results['methods'] == "KLD"].max()['aucs'], 
            results[results['methods'] == "Random"].max()['aucs'], 
            results[results['methods'] == "LSV"].max()['aucs']]

    
    for i in range(4):
        plt.text(x=bp.get_xticks()[i+1] - 0.07, y=top[i] + 0.001, s='**', fontdict={'size':12, 'color':'black'})


    ax.set_ylabel('AUC')
    
    if method == "retrain": # not for bottom row
        ax.set_xlabel('Pathway Ranking Method')
    else: 
         ax.set_xlabel('')
    
    #plt.title(get_title(dataset) + ' ' + method.capitalize() + ' Benchmark')
    plt.title(method.capitalize())

    
    plt.savefig('figs/dataset=%s-method=%s.pdf' % (dataset, method), bbox_inches='tight')

    plt.show()
    
    
    
def get_title(dataset): 
    dataset_title = ''
    if dataset == 'kang':
        dataset_title = 'PBMC'
    if dataset == 'haber':
        dataset_title = 'Intestinal'  
    if dataset == 'datlinger':
        dataset_title = 'Jurkat' 
    if dataset == 'grubman':
        dataset_title = 'Entorhinal'   
    return dataset_title 
    
    

# get single line graph 
def get_lines(dataset, method): 
    ig_results, logvar_results, lr_results, kld_results, rand_results = get_arrays(dataset, method)
    
    plt.style.use('seaborn-colorblind')
    
    fig, ax = plt.subplots(figsize=(6,4))

    sb.lineplot(data=ig_results.mean(0), label='PAUSE')
    sb.lineplot(data=lr_results.mean(0), label='LR')
    sb.lineplot(data=kld_results.mean(0), label='KLD')
    sb.lineplot(data=rand_results.mean(0), label='Random')
    sb.lineplot(data=logvar_results.mean(0), label='LSV')

    
    if method == 'impute': 
        ax.set_xlabel('Number of Top Pathways Ablated')

    if method == 'retrain': 
        ax.set_xlabel('Number of Top Pathways Included')
    
    plt.legend(fontsize=LEG_SIZE)
    
    ax.set_ylabel('Reconstruction Error')
    
    #plt.title(get_title(dataset) + ' ' + method.capitalize() + ' Benchmark')
    plt.title(method.capitalize())

    
    plt.savefig('figs/lines-dataset=%s-method=%s.pdf' % (dataset, method),bbox_inches='tight')

    plt.show()


if __name__ == '__main__':
    
    for dataset in DATASETS: 
        for method in METHODS: 
            get_subplot(dataset, method) 
    
    get_lines('haber', 'impute')
    get_lines('haber', 'retrain')
    
