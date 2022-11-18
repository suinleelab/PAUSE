import pandas as pd
import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import NegativeBinomial, Normal, Distribution, Gamma, Poisson, constraints
from torch.distributions import kl_divergence as kl
from torch.distributions.utils import (
    broadcast_all,
    lazy_property,
    logits_to_probs,
    probs_to_logits,
)

import warnings
from typing import Optional, Tuple, Union

from scipy.linalg import block_diag
from collections import namedtuple
Outputs = namedtuple('Outputs', 'z global_recon module_outputs mu logvar')
OutputsNB = namedtuple('Outputs', 'z global_recon mu logvar output_mean')
VAEOutputs = namedtuple('VAEOutputs', 'z global_recon mu logvar')

#utility function for kernel-based losses
def gram_matrix(x, sigma=1):
    pairwise_distances = x.unsqueeze(1) - x
    return torch.exp(-pairwise_distances.norm(2, dim=2) / (2 * sigma * sigma))

class vaeEncoder(nn.Sequential):
    ##
    ## Encoder Module for a standard VAE
    ##
    def __init__(self,
        n_features,
        hidden_layers,
        activation='elu',
        batch_norm=True,
        cdim=None,
        **kwargs):
        super(vaeEncoder, self).__init__()
        
        for i in list(range(len(hidden_layers))):
            if i == 0:
                module_name = 'encoder_dense_%d' % (i + 1)
                self.add_module(module_name, nn.Linear(n_features,hidden_layers[i]))
                if batch_norm:
                    module_name = 'encoder_norm_%d' % (i + 1)
                    self.add_module(module_name, nn.BatchNorm1d(hidden_layers[i])),
                module_name = 'encoder_elu_%d' % (i + 1)
                self.add_module(module_name, nn.ELU(inplace=True))
            elif i == (len(hidden_layers) - 1):
                module_name = 'encoder_dense_%d' % (i + 1)
                self.add_module(module_name, nn.Linear(hidden_layers[i-1],hidden_layers[i]*2))
            else:
                module_name = 'encoder_dense_%d' % (i + 1)
                self.add_module(module_name, nn.Linear(hidden_layers[i-1],hidden_layers[i]))
                if batch_norm:
                    module_name = 'encoder_norm_%d' % (i + 1)
                    self.add_module(module_name, nn.BatchNorm1d(hidden_layers[i])),
                module_name = 'encoder_elu_%d' % (i + 1)
                self.add_module(module_name, nn.ELU(inplace=True))
                
class vaeDecoder(nn.Sequential):
    ##
    ## Decoder Module for standard VAE
    ##
    def __init__(self,
        n_features,
        hidden_layers,
        activation='elu',
        batch_norm=True,
        cdim=None,
        **kwargs):
        super(vaeDecoder, self).__init__()
        
        reversedLayersList = hidden_layers[::-1]
        
        for i in list(range(len(reversedLayersList))):
            if i == (len(hidden_layers) - 1):
                module_name = 'decoder_dense_%d' % (i + 1)
                self.add_module(module_name, nn.Linear(reversedLayersList[i],n_features))
            else:
                module_name = 'decoder_dense_%d' % (i + 1)
                self.add_module(module_name, nn.Linear(reversedLayersList[i],reversedLayersList[i+1]))
                if batch_norm:
                    module_name = 'decoder_norm_%d' % (i + 1)
                    self.add_module(module_name, nn.BatchNorm1d(reversedLayersList[i+1])),
                module_name = 'decoder_elu_%d' % (i + 1)
                self.add_module(module_name, nn.ELU(inplace=True))

class linearDecoder(nn.Sequential):
    ##
    ## Linear decoder module for a VAE
    ##
    def __init__(self,
        n_features,
        hidden_layers,
        activation='elu',
        batch_norm=True,
        cdim=None,
        **kwargs):
        super(linearDecoder, self).__init__()
        
        reversedLayersList = hidden_layers[::-1]
        module_name = 'linear_decoder_dense'
        self.add_module(module_name, nn.Linear(reversedLayersList[0],n_features))

class VAE(nn.Module):
    ##
    ## Combined VAE Module
    ## forward method returns z, global_recon, mu, logvar
    ##
    def __init__(self, n_features, hidden_layers,
                 activation='elu',
                 batch_norm=True,
                 decoder='neural',
                 use_gpu=True,
                 **kwargs):
        super(VAE, self).__init__()
        
        self.n_features = n_features
        self.hidden_layers = hidden_layers
        self.use_gpu = use_gpu
        
        self.encoder_net = vaeEncoder(n_features,
                        hidden_layers,
                        activation=activation,
                        batch_norm=batch_norm,
                        **kwargs)
        if decoder == 'neural':
            self.decoder_net = vaeDecoder(n_features,
                            hidden_layers,
                            activation=activation,
                            batch_norm=batch_norm,
                            **kwargs)
        elif decoder == 'linear':
            self.decoder_net = linearDecoder(n_features,
                            hidden_layers,
                            activation=activation,
                            batch_norm=batch_norm,
                            **kwargs)
    
    def encode(self, x, **kwargs):
        params = self.encoder_net(x, **kwargs)
        mu, logvar = torch.split(params, int(params.size(1)/2), dim=1)
        return mu, logvar

    def decode(self, z, **kwargs):
        module_outputs = self.decoder_net(z, **kwargs)
        return module_outputs
    
    def reparametrize(self, mu, logvar):
        if self.use_gpu:
            eps = torch.randn(logvar.shape).cuda()
        else:
            eps = torch.randn(logvar.shape)
        return mu + torch.exp(logvar / 2) * eps
    
    def forward(self, x, c=None, **kwargs):
        
        if c is not None:
            network_input = torch.cat([x, c], 1)
        else:
            network_input = x
            
        mu, logvar = self.encode(network_input, **kwargs)
        z = self.reparametrize(mu, logvar)
        
        if c is not None:
            latent_input = torch.cat([z, c], 1)
        else:
            latent_input = z
        
        global_recon = self.decode(latent_input, **kwargs)
        outputs = VAEOutputs(z, global_recon, mu, logvar)
            
        return outputs

class VAEModel(object):
    'VAE Model class with training methods'
    def __init__(self, 
                 n_features, 
                 hidden_layers,
                 beta=1e-5,
                 activation='elu',
                 batch_norm=True,
                 use_gpu=True,
                 decoder='neural',
                 **kwargs):
        '''
        initialize model.
        '''
        self.model = VAE(n_features, hidden_layers,
                 activation='elu',
                 batch_norm=True,
                 decoder=decoder,
                 use_gpu=use_gpu,
                 **kwargs)
        self.use_gpu = use_gpu
        self.beta = beta
        
        if self.use_gpu:
            self.model.cuda()
            
            
    def loss_function(self, recon_x, x, mu, log_var, val=False):
        MSE = F.mse_loss(recon_x, x.view(-1, recon_x.size(1)), reduction='sum')
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        if val:
            return MSE
        else:
            return MSE + self.beta * KLD
    
    def train(self, 
              train_dataset, 
              val_dataset, 
              max_epochs=100,
              lr=0.001, 
              weight_decay=1e-4,
              batch_size=16,
              logpath=None,
              checkpoint_path='checkpoint.pkl',
              verbose=True):
        self.checkpoint_path = checkpoint_path
        self.lr = lr
        self.weight_decay = weight_decay
        
        # Create torch DataLoaders from the training and validation datasets.
        train_dataloader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=2)
        val_dataloader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=2)
        
        self.n_features = train_dataset.X.shape[1]
        
        self.optimizer = optim.Adam(self.model.parameters(),lr=self.lr)
        
        best_loss = None
        for i_epoch in range(max_epochs):
            print("-------- Epoch {:03d} --------".format(i_epoch))
            
            trainloss = self._train_epoch(train_dataloader)
            trainloss /= len(train_dataset)
            valloss = self._val_epoch(val_dataloader)
            valloss /= len(val_dataset)
            
            # only save if improvement
            if best_loss is None or valloss < best_loss: 
                best_loss = valloss
                self._checkpoint(i_epoch, valloss, suffix='.best_loss')
            else:
                self.lr = self.lr/10.
                self.optimizer = optim.Adam(self.model.parameters(),lr=self.lr)
            # Write information on this epoch to a log.
            logstr = "Epoch {:03d}: ".format(i_epoch) +\
                     "training loss {:08.4f},".format(trainloss) +\
                     "validation loss {:08.4f}".format(valloss)
            if not logpath is None:
                with open(logpath, 'a') as logfile:
                    logfile.write(logstr + '\n')
            if verbose:
                print(logstr)
        self.load_checkpoint(self.checkpoint_path+'.best_loss')
        
    def _train_epoch(self,train_dataloader,use_c=False):
        self.model.train()
        train_loss = 0
        for batch_idx, data in enumerate(train_dataloader):
            if use_c:
                if self.use_gpu:
                    x, c = data[0].float().cuda(), data[1].float().cuda()
                else:
                    x, c = data[0].float(), data[1].float()
            
            else:
                if self.use_gpu:
                    x = data.float().cuda()
                else:
                    x = data.float()
                    
            self.optimizer.zero_grad()

            outputs = self.model(x)
            loss = self.loss_function(outputs.global_recon, x, outputs.mu, outputs.logvar)

            loss.backward()
            train_loss += loss.item()
            self.optimizer.step()
        return train_loss
    
    def _val_epoch(self,val_dataloader,use_c=False):
        self.model.train(False)
        val_loss = 0
        for batch_idx, data in enumerate(val_dataloader):
            if use_c:
                if self.use_gpu:
                    x, c = data[0].float().cuda(), data[1].float().cuda()
                else:
                    x, c = data[0].float(), data[1].float()
            
            else:
                if self.use_gpu:
                    x = data.float().cuda()
                else:
                    x = data.float()

            outputs = self.model(x)
            loss = self.loss_function(outputs.global_recon, x, outputs.mu, outputs.logvar, val=True)

            val_loss += loss.item()
        return val_loss
    
    def _checkpoint(self, epoch, valloss, suffix=None):
        '''
        Save a checkpoint to self.checkpoint_path, including the full model, 
        current epoch, learning rate, and random number generator state.
        '''
        state = {'model': self.model,
                 'best_loss': valloss,
                 'epoch': epoch,
                 'rng_state': torch.get_rng_state(),
                 'LR': self.lr ,
                 'optimizer': self.optimizer.state_dict()}
        checkpoint_path = self.checkpoint_path
        if suffix is not None:
            checkpoint_path = checkpoint_path + suffix
        torch.save(state, checkpoint_path)
        
    def get_recon_error(self, 
                        val_dataset,
                        batch_size=256):
        val_dataloader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=2)
        
        valloss = self._val_epoch(val_dataloader)
        valloss /= len(val_dataset)
        
        return valloss
        
    def load_checkpoint(self, path, load_optimizer=False):
        checkpoint = torch.load(path)
        self.model = checkpoint['model']
        if load_optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            
    def set_gpu(self, use_gpu):
        assert type(use_gpu) is bool, 'Argument must be "True" or "False"'
        if use_gpu:
            self.model.cuda()
            self.model.use_gpu = use_gpu
            self.use_gpu = use_gpu
            
        else:
            self.model.cpu()
            self.model.use_gpu = use_gpu
            self.use_gpu = use_gpu

#####
##### Layers and Support Functions for Pathway Sparse VAE
#####

#################################
# Define custom autograd function for masked connection.

class CustomizedLinearFunction(torch.autograd.Function):
    """
    autograd function which masks it's weights by 'mask'.
    code from: https://github.com/uchida-takumi/CustomizedLinear
    """

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias, mask is an optional argument
    def forward(ctx, input, weight, bias=None, mask=None):
        if mask is not None:
            # change weight to 0 where mask == 0
            weight = weight * mask
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        ctx.save_for_backward(input, weight, bias, mask)
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        input, weight, bias, mask = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = grad_mask = None

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
            if mask is not None:
                # change grad_weight to 0 where mask == 0
                grad_weight = grad_weight * mask
        #if bias is not None and ctx.needs_input_grad[2]:
        if ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_bias, grad_mask


class CustomizedLinear(nn.Module):
    def __init__(self, mask, bias=True):
        """
        extended torch.nn module which masks connections.
        Arguments
        ------------------
        mask [torch.tensor]:
            the shape is (n_input_feature, n_output_feature).
            the elements are 0 or 1 which declare un-connected or
            connected.
        bias [bool]:
            flag of bias.
        """
        super(CustomizedLinear, self).__init__()
        self.input_features = mask.shape[0]
        self.output_features = mask.shape[1]
        if isinstance(mask, torch.Tensor):
            self.mask = mask.type(torch.float).t()
        else:
            self.mask = torch.tensor(mask, dtype=torch.float).t()

        self.mask = nn.Parameter(self.mask, requires_grad=False)

        # nn.Parameter is a special kind of Tensor, that will get
        # automatically registered as Module's parameter once it's assigned
        # as an attribute. Parameters and buffers need to be registered, or
        # they won't appear in .parameters() (doesn't apply to buffers), and
        # won't be converted when e.g. .cuda() is called. You can use
        # .register_buffer() to register buffers.
        # nn.Parameters require gradients by default.
        self.weight = nn.Parameter(torch.Tensor(self.output_features, self.input_features))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.output_features))
        else:
            # You should always register all possible parameters, but the
            # optional ones can be None if you want.
            self.register_parameter('bias', None)
        self.reset_parameters()

        # mask weight
        self.weight.data = self.weight.data * self.mask

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)


    def forward(self, input):
        # See the autograd section for explanation of what happens here.
        return CustomizedLinearFunction.apply(input, self.weight, self.bias, self.mask)

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return 'input_features={}, output_features={}, bias={}'.format(
            self.input_features, self.output_features, self.bias is not None
        )

##
## torch implementation of https://github.com/ratschlab/pmvae
##

def build_module_connector(membership_mask, first_layer_nodes):
    '''Builds a mask to connect the input (genes) to the pathway
    modules by repeating the membership_mask (which maps genes
    to pathways) for each node in the first hidden layer of the
    pathway modules.
    
    membership_mask: bool nparray, shape pathways x genes
    first_layer_nodes: int, number of nodes in first module layer
    
    returns the module connector mask, which is a boolean array
    with 2 dimensions n_genes x (n_pathways * first_layer_nodes)
    '''
    
    ## consider making the membership mask transposed
    ## and then just changing the code so it says axis = 1
    ## and no transpose at the end
    return np.repeat(membership_mask, first_layer_nodes, axis=0).T

def build_module_isolation_mask(nmodules, module_output_dim):
    '''Isolates a single module for gradient steps
    Used for the local reconstruction terms, drops all modules except one
    '''
    blocks = [np.ones((1, module_output_dim))] * nmodules
    return block_diag(*blocks)

def build_separation_mask(input_dim, output_dim, nmodules):
    '''Builds a large block diagonal matrix for subsequent
    "dense" layers of the neural network to maintain dense
    connections within modules and no connections between modules.
    '''
    ##
    ## this code could definitely be modified to first append
    ## dense connections of one size for pathway blocks
    ## then append connections of another size for cell type 
    ## blocks
    ##
    blocks = [np.ones((input_dim, output_dim))] * nmodules
    return block_diag(*blocks)

def build_mask_list(membership_mask, hidden_layers, latent_dim):
    '''Builds the masks used by encoders/decoders
    membership_mask: boolean array, modules x features
    hidden_layers: width of each hidden layer (list of ints)
    latent_dim: size of each module latent dim
    pathway mask assigns genes to pathway modules
    separation masks keep modules separated
    Encoder modifies the last separation mask to give mu/logvar
    Decoder reverses and transposes the masks
    '''
    
    ##
    ## if we transpose the membership mask, this will need to flip as well
    nmodules, nfeats = membership_mask.shape
    base = list()
    
    #
    # first build the module connector mask, which maps from genes to modules 
    base.append(build_module_connector(membership_mask, hidden_layers[0]))
    dims = hidden_layers + [latent_dim]
    #
    # then for each additional layer, build the separation mask
    for dinput, doutput in zip(dims[:-1], dims[1:]):
        base.append(build_separation_mask(dinput, doutput, nmodules))

    base = [mask.astype(np.float32) for mask in base]
    return base

class pmEncoder(nn.Sequential):
    ##
    ## Encoder Module for pmVAE
    ##
    def __init__(self, 
        membership_mask,
        hidden_layers,
        latent_dim,
        activation='elu',
        batch_norm=True,
        cdim=None,
        unsupervised=True,
        **kwargs):
        super(pmEncoder, self).__init__()
        
        self.masks = build_mask_list(membership_mask, hidden_layers, latent_dim)
        
        # if you're adding conditions, add cdim extra columns to map those features
        # to all modules
        if cdim is not None:
            self.masks[0] = np.vstack(
                    (self.masks[0], np.ones((cdim,self.masks[0].shape[1]))))
            
        if unsupervised:
            # mask for mu and mask for logvar
            self.masks[-1] = np.hstack((self.masks[-1], self.masks[-1]))
        
        for i,mask in enumerate(self.masks[:-1]):
            module_name = 'encoder_dense_%d' % (i + 1)
            self.add_module(module_name, CustomizedLinear(mask))
            if batch_norm:
                module_name = 'encoder_norm_%d' % (i + 1)
                self.add_module(module_name, nn.BatchNorm1d(mask.shape[1])),
            module_name = 'encoder_elu_%d' % (i + 1)
            self.add_module(module_name, nn.ELU(inplace=True))
        module_name = 'encoder_dense_%d' % (i + 2)
        self.add_module(module_name, CustomizedLinear(self.masks[-1]))
        if batch_norm:
            module_name = 'encoder_norm_%d' % (i + 2)
            self.add_module(module_name, nn.BatchNorm1d(self.masks[-1].shape[1])),
            
class pmDecoder(nn.Sequential):
    ##
    ## Decoder Module for pmVAE
    ##
    def __init__(self, 
        membership_mask,
        hidden_layers,
        latent_dim,
        activation='elu',
        batch_norm=True,
        cdim=None,
        **kwargs):
        super(pmDecoder, self).__init__()
        
        self.masks = build_mask_list(membership_mask, hidden_layers, latent_dim)
        # transpose masks for decoding
        self.masks = [mask.T for mask in self.masks[::-1]]
        
        # if you're adding conditions, add cdim extra rows to map those features
        # to all modules
        if cdim is not None:
            self.masks[0] = np.vstack(
                    (self.masks[0], np.ones((cdim,self.masks[0].shape[1]))))
        
        for i,mask in enumerate(self.masks[:-1]):
            module_name = 'decoder_dense_%d' % (i + 1)
            self.add_module(module_name, CustomizedLinear(mask))
            if batch_norm:
                module_name = 'decoder_norm_%d' % (i + 1)
                self.add_module(module_name, nn.BatchNorm1d(mask.shape[1])),
            module_name = 'decoder_elu_%d' % (i + 1)
            self.add_module(module_name, nn.ELU(inplace=True))
            
class pmVAE(nn.Module):
    ##
    ## Full pmVAE model
    ##
    def __init__(self, 
        membership_mask,
        hidden_layers,
        latent_dim,
        activation='elu',
        batch_norm=True,
        decoder='neural',
        bias_last_layer=False,
        add_auxiliary_module=False,
        cdim=None,
        terms=None,
        use_gpu=True,
        **kwargs):
        super(pmVAE, self).__init__()
        
        self.decoder=decoder
        
        self.use_gpu = use_gpu
        
        self.num_annotated_modules, self.num_feats = membership_mask.shape
        if isinstance(membership_mask, pd.DataFrame):
            terms = membership_mask.index
            membership_mask = membership_mask.values
        
        self.add_auxiliary_module = add_auxiliary_module
        if add_auxiliary_module:
            membership_mask = np.vstack(
                    (membership_mask, np.ones_like(membership_mask[0])))
            if terms is not None:
                terms = list(terms) + ['AUXILIARY']
                
        self.cdim = cdim
                
        self.membership_mask=membership_mask
        self.module_isolation_mask = build_module_isolation_mask(
                self.membership_mask.shape[0],
                hidden_layers[-1])
        
        self._module_latent_dim = latent_dim
        self._hidden_layers = hidden_layers
        assert len(terms) == len(self.membership_mask)
        self.terms = list(terms)
        
        self.encoding_masks = build_mask_list(membership_mask, hidden_layers, latent_dim)
#         # transpose masks for decoding
        self.decoding_masks = [mask.T for mask in self.encoding_masks[::-1]]
        if cdim is not None:
            self.encoding_masks[0] = np.vstack(
                    (self.encoding_masks[0], np.ones((cdim,self.encoding_masks[0].shape[1]))))
            self.decoding_masks[0] = np.vstack(
                    (self.decoding_masks[0], np.ones((cdim,self.decoding_masks[0].shape[1]))))
        
        self.encoder_net = pmEncoder(membership_mask,
                        hidden_layers,
                        latent_dim,
                        activation='elu',
                        batch_norm=True,
                        cdim=cdim,
                        **kwargs)
        if decoder == 'neural':
            self.decoder_net = pmDecoder(membership_mask,
                            hidden_layers,
                            latent_dim,
                            activation='elu',
                            batch_norm=True,
                            cdim=cdim,
                            **kwargs)
        elif decoder == 'linear':
            self.decoder_net = linearDecoder(membership_mask.shape[1],
                            hidden_layers,
                            activation='elu',
                            batch_norm=True,
                            cdim=cdim,
                            **kwargs)

        self.merge_layer = CustomizedLinear(self.decoding_masks[-1],bias=bias_last_layer)
        
    def encode(self, x, **kwargs):
        params = self.encoder_net(x, **kwargs)
        mu, logvar = torch.split(params, int(params.size(1)/2), dim=1)
        return mu, logvar

    def decode(self, z, **kwargs):
        module_outputs = self.decoder_net(z, **kwargs)
        global_recon = self.merge(module_outputs, **kwargs)
        return global_recon

    def merge(self, module_outputs, **kwargs):
        global_recon = self.merge_layer(module_outputs, **kwargs)
        return global_recon
    
    def reparametrize(self, mu, logvar):
        if self.use_gpu:
            eps = torch.randn(logvar.shape).cuda()
        else:
            eps = torch.randn(logvar.shape)
        return mu + torch.exp(logvar / 2) * eps
    
    def get_masks_for_local_losses(self):
        if self.add_auxiliary_module:
            return zip(self.membership_mask[:-1],
                       self.module_isolation_mask[:-1])

        return zip(self.membership_mask,
                   self.module_isolation_mask)
    
    def forward(self, x, c=None, **kwargs):
                
        if c is not None:
            network_input = torch.cat([x, c], 1)
        else:
            network_input = x
            
        mu, logvar = self.encode(network_input, **kwargs)
        z = self.reparametrize(mu, logvar)
        
        if c is not None:
            latent_input = torch.cat([z, c], 1)
        else:
            latent_input = z
        
        module_outputs = self.decoder_net(latent_input, **kwargs)
        global_recon = self.merge(module_outputs, **kwargs)
        outputs = Outputs(z, global_recon, module_outputs, mu, logvar)
            
        return outputs
    
class pmVAEModel(object):
    'A full model training wrapper for the pmVAE model'
    def __init__(self, 
        membership_mask,
        hidden_layers,
        latent_dim,
        cdim = None,
        hsic_penalty=None,
        activation='elu',
        batch_norm=True,
        bias_last_layer=False,
        add_auxiliary_module=False,
        use_gpu=True,
        **kwargs):
        '''
        Create a pmVAE for rna-seq.
        '''
        super(pmVAEModel, self).__init__()
        
        self.cdim = cdim
        self.hsic_penalty = hsic_penalty
        
        self.model = pmVAE( 
        membership_mask,
        hidden_layers,
        latent_dim,
        cdim=cdim,
        activation=activation,
        batch_norm=batch_norm,
        bias_last_layer=bias_last_layer,
        add_auxiliary_module=add_auxiliary_module,
        use_gpu=use_gpu,
        **kwargs)
        
        self.use_gpu=use_gpu
        
        if self.use_gpu:
            self.model.cuda()
            
    def weighted_mse(self, y_true, y_pred, sample_weight):
        if self.use_gpu:
            sample_weight = torch.tensor(sample_weight, dtype=y_pred.dtype).cuda()
        else:
            sample_weight = torch.tensor(sample_weight, dtype=y_pred.dtype)
        diff = torch.pow(y_true - y_pred, 2) * sample_weight
        wmse = torch.sum(diff) / torch.sum(sample_weight)
        return wmse
    
    def compute_hsic(self, x, y, sigma=1):
        m = x.shape[0]
        K = gram_matrix(x, sigma=sigma)
        L = gram_matrix(y, sigma=sigma)
        H = torch.eye(m) - 1.0 / m * torch.ones((m, m))
        if self.use_gpu:
            H = H.float().cuda()
        else:
            H = H.float()
        HSIC = torch.trace(torch.mm(L, torch.mm(H, torch.mm(K, H)))) / ((m - 1) ** 2)
        return HSIC
        
    def calc_loss(self, data, pathway_dropout=True, val=False, use_c=False):
        
        if use_c:
            if self.use_gpu:
                x, c = data[0].float().cuda(), data[1].float().cuda()
            else:
                x, c = data[0].float(), data[1].long()
            outputs = self.model(x,c)
            
        else:
            if self.use_gpu:
                x = data.float().cuda()
            else:
                x = data.float()
            outputs = self.model(x)
        
        MSE = F.mse_loss(outputs.global_recon, x.view(-1, self.x_dim), reduction='sum')
        
        if val:
            return MSE
        else:
            KLD = -0.5 * torch.sum(1 + outputs.logvar - outputs.mu.pow(2) - outputs.logvar.exp())

            local_recon_loss = 0.0
            
            if pathway_dropout:
                for feat_mask, module_mask in self.model.get_masks_for_local_losses():
                    if self.use_gpu:
                        feat_mask, module_mask = torch.tensor(feat_mask).float().cuda(), torch.tensor(module_mask).float().cuda()
                    else:
                        feat_mask, module_mask = torch.tensor(feat_mask).float(), torch.tensor(module_mask).float()
                    # dropout other modules & reconstruct
                    only_active_module = torch.mul(outputs.module_outputs, module_mask)
                    local_recon = self.model.merge(only_active_module)

                    # only compute the loss with participating genes
                    wmse = self.weighted_mse(x, local_recon, feat_mask)

                    local_recon_loss = local_recon_loss + wmse

                local_recon_loss = local_recon_loss / self.model.num_annotated_modules
            
            else:
                local_recon_loss = 0.0
            
            if self.hsic_penalty is not None:
                hsic_loss = self.compute_hsic(outputs.z, c)
                full_loss = MSE + local_recon_loss + self.beta * KLD + self.hsic_penalty * hsic_loss
            else:
                full_loss = MSE + local_recon_loss + self.beta * KLD
            
            return full_loss
        
    def get_recon_error(self, 
                        val_dataset,
                        batch_size=256):
        val_dataloader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=2)
        
        valloss = self._val_epoch(val_dataloader)
        valloss /= len(val_dataset)
        
        return valloss
        
        
    
    def train(self, 
              train_dataset, 
              val_dataset, 
              max_epochs=1200,
              lr=0.001,
              beta=1e-5,
              batch_size=256,
              pathway_dropout=True,
              logpath=None,
              checkpoint_path='pmvae_checkpoint.pkl',
              verbose=True):
        self.checkpoint_path = checkpoint_path
        self.lr = lr
        self.beta = beta
        
        # Create torch DataLoaders from the training and validation datasets.
        # Necessary for batching and shuffling data.
        train_dataloader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=2)
        val_dataloader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=2)
        
        self.x_dim = train_dataset.X.shape[1]
        
        self.optimizer = optim.Adam(self.model.parameters(),lr=self.lr)
        
        best_loss = None
        for i_epoch in range(max_epochs):
            print("-------- Epoch {:03d} --------".format(i_epoch))
            
            if self.cdim is not None:
                use_c=True
            else:
                use_c=False
                
            trainloss = self._train_epoch(train_dataloader, pathway_dropout=pathway_dropout, use_c=use_c)
            trainloss /= len(train_dataset)
            valloss = self._val_epoch(val_dataloader, use_c=use_c)
            valloss /= len(val_dataset)
            
            # only save if improvement
            if best_loss is None or valloss < best_loss: 
                best_loss = valloss
                self._checkpoint(i_epoch, valloss, suffix='.best_loss')
            else:
                self.lr = self.lr/10.
                self.optimizer = optim.Adam(self.model.parameters(),lr=self.lr)
            # Write information on this epoch to a log.
            logstr = "Epoch {:03d}: ".format(i_epoch) +\
                     "training loss {:08.4f},".format(trainloss) +\
                     "validation loss {:08.4f}".format(valloss)
            if not logpath is None:
                with open(logpath, 'a') as logfile:
                    logfile.write(logstr + '\n')
            if verbose:
                print(logstr)
        self.load_checkpoint(self.checkpoint_path+'.best_loss')
        
    def _train_epoch(self,train_dataloader,pathway_dropout=True,use_c=False):
        self.model.train()
        train_loss = 0
        for batch_idx, data in enumerate(train_dataloader):
            self.optimizer.zero_grad()
            loss = self.calc_loss(data, pathway_dropout=pathway_dropout, use_c=use_c)
            loss.backward()
            train_loss += loss.item()
            self.optimizer.step()
        return train_loss
    
    def _val_epoch(self,val_dataloader,use_c=False):
        self.model.train(False)
        val_loss = 0
        for batch_idx, data in enumerate(val_dataloader):
            loss = self.calc_loss(data, val=True, use_c=use_c)
            val_loss += loss.item()
        return val_loss
    
    def _checkpoint(self, epoch, valloss, suffix=None):
        '''
        Save a checkpoint to self.checkpoint_path, including the full model, 
        current epoch, learning rate, and random number generator state.
        '''
        state = {'model': self.model,
                 'best_loss': valloss,
                 'epoch': epoch,
                 'rng_state': torch.get_rng_state(),
                 'LR': self.lr ,
                 'optimizer': self.optimizer.state_dict()}
        checkpoint_path = self.checkpoint_path
        if suffix is not None:
            checkpoint_path = checkpoint_path + suffix
        torch.save(state, checkpoint_path)
    def load_checkpoint(self, path, load_optimizer=False):
        checkpoint = torch.load(path)
        self.model = checkpoint['model']
        if load_optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            
    def latent_space_names(self, terms=None):
        terms = self.model.terms if terms is None else terms
        assert terms is not None, 'Need to specify gene set terms'

        if self.model.add_auxiliary_module \
                and len(terms) == self.model.num_annotated_modules:
            terms = list(terms) + ['AUXILIARY']

        z = self.model._module_latent_dim
        repeated_terms = np.repeat(terms, z)
        index = np.tile(range(z), len(terms)).astype(str)
        latent_dim_names = map('-'.join, zip(repeated_terms, index))

        return list(latent_dim_names)
    
    def set_gpu(self, use_gpu):
        assert type(use_gpu) is bool, 'Argument must be "True" or "False"'
        if use_gpu:
            self.model.cuda()
            self.model.use_gpu = use_gpu
            self.use_gpu = use_gpu
            
        else:
            self.model.cpu()
            self.model.use_gpu = use_gpu
            self.use_gpu = use_gpu
            
######
###### pmVAE with Neg binomial dist
######

class pmVAEnb(nn.Module):
    ##
    ## Full pmVAE model that fits a neg binomial dist in the last layer
    ## Negative binomial generative model adapted from
    ## tutorial here: https://docs.scvi-tools.org/en/stable/tutorials/notebooks/module_user_guide.html
    ##
    def __init__(self, 
        membership_mask,
        hidden_layers,
        latent_dim,
        activation='elu',
        batch_norm=True,
        decoder='neural',
        bias_last_layer=False,
        add_auxiliary_module=False,
        cdim=None,
        terms=None,
        use_gpu=True,
        **kwargs):
        super(pmVAEnb, self).__init__()
        
        self.decoder=decoder
        
        self.use_gpu = use_gpu
        
        self.num_annotated_modules, self.num_feats = membership_mask.shape
        if isinstance(membership_mask, pd.DataFrame):
            terms = membership_mask.index
            membership_mask = membership_mask.values
        
        self.add_auxiliary_module = add_auxiliary_module
        if add_auxiliary_module:
            membership_mask = np.vstack(
                    (membership_mask, np.ones_like(membership_mask[0])))
            if terms is not None:
                terms = list(terms) + ['AUXILIARY']
                
        self.cdim = cdim
                
        self.membership_mask=membership_mask
        self.module_isolation_mask = build_module_isolation_mask(
                self.membership_mask.shape[0],
                hidden_layers[-1])
        
        self._module_latent_dim = latent_dim
        self._hidden_layers = hidden_layers
        assert len(terms) == len(self.membership_mask)
        self.terms = list(terms)
        
        self.encoding_masks = build_mask_list(membership_mask, hidden_layers, latent_dim)
#         # transpose masks for decoding
        self.decoding_masks = [mask.T for mask in self.encoding_masks[::-1]]
        if cdim is not None:
            self.encoding_masks[0] = np.vstack(
                    (self.encoding_masks[0], np.ones((cdim,self.encoding_masks[0].shape[1]))))
            self.decoding_masks[0] = np.vstack(
                    (self.decoding_masks[0], np.ones((cdim,self.decoding_masks[0].shape[1]))))
        
        self.encoder_net = pmEncoder(membership_mask,
                        hidden_layers,
                        latent_dim,
                        activation='elu',
                        batch_norm=True,
                        cdim=cdim,
                        **kwargs)
        if decoder == 'neural':
            self.decoder_net = pmDecoder(membership_mask,
                            hidden_layers,
                            latent_dim,
                            activation='elu',
                            batch_norm=True,
                            cdim=cdim,
                            **kwargs)
        elif decoder == 'linear':
            self.decoder_net = linearDecoder(membership_mask.shape[1],
                            hidden_layers,
                            activation='elu',
                            batch_norm=True,
                            cdim=cdim,
                            **kwargs)

        self.merge_layer = CustomizedLinear(self.decoding_masks[-1],bias=bias_last_layer)
        
        self.log_theta = torch.nn.Parameter(torch.randn(membership_mask.shape[1]))
        
    def encode(self, x, **kwargs):
        params = self.encoder_net(x, **kwargs)
        mu, logvar = torch.split(params, int(params.size(1)/2), dim=1)
        return mu, logvar

    def decode(self, z, **kwargs):
        module_outputs = self.decoder_net(z, **kwargs)
        global_recon = self.merge(module_outputs, **kwargs)
        return global_recon

    def merge(self, module_outputs, **kwargs):
        global_recon = self.merge_layer(module_outputs, **kwargs)
        return global_recon
    
    def generative(self, z, library):
        """Runs the generative model."""

        # get the "normalized" mean of the negative binomial
        px_scale = torch.nn.Softmax(dim=-1)(self.decode(z))
        # get the mean of the negative binomial
        px_rate = library * px_scale
        # get the dispersion parameter
        theta = torch.exp(self.log_theta)

        return dict(
            px_scale=px_scale, theta=theta, px_rate=px_rate
        )
    
    def reparametrize(self, mu, logvar):
        if self.use_gpu:
            eps = torch.randn(logvar.shape).cuda()
        else:
            eps = torch.randn(logvar.shape)
        return mu + torch.exp(logvar / 2) * eps
    
    def get_masks_for_local_losses(self):
        if self.add_auxiliary_module:
            return zip(self.membership_mask[:-1],
                       self.module_isolation_mask[:-1])

        return zip(self.membership_mask,
                   self.module_isolation_mask)
    
    def calc_likelihood_latent_z(self, x, z, c=None, **kwargs):
        library = torch.sum(x, dim=1, keepdim=True)
   
        x_ = torch.log(1.0 + x)
                
        if c is not None:
            network_input = torch.cat([x_, c], 1)
        else:
            network_input = x_
        mu, logvar = self.encode(network_input, **kwargs)
        generative_params = self.generative(z, library)
        px_rate = generative_params["px_rate"]
        theta = generative_params["theta"]
        qz_m = mu
        qz_v = torch.exp(logvar / 2)
        nb_logits = (px_rate + 1e-4).log() - (theta + 1e-4).log()
        log_lik = NegativeBinomial(theta, logits=nb_logits).log_prob(x).sum(dim=-1)
        return -log_lik.view(-1,1)
        
    def forward(self, x, c=None, **kwargs):
        
        # log the input to the variational distribution for numerical stability
        x_ = torch.log(1.0 + x)
                
        if c is not None:
            network_input = torch.cat([x_, c], 1)
        else:
            network_input = x_
            
        mu, logvar = self.encode(network_input, **kwargs)
        z = self.reparametrize(mu, logvar)
        
        if c is not None:
            latent_input = torch.cat([z, c], 1)
        else:
            latent_input = z
        
        module_outputs = self.decoder_net(latent_input, **kwargs)
        px_scale = self.merge(module_outputs, **kwargs)
        
        library = torch.sum(x, dim=1, keepdim=True)
        
        generative_params = self.generative(z, library)
        px_rate = generative_params["px_rate"]
        theta = generative_params["theta"]
        qz_m = mu
        qz_v = logvar
        
        # term 1
        # the pytorch NB distribution uses a different parameterization
        # so we must apply a quick transformation (included in scvi-tools, but here we use the pytorch code)
        nb_logits = (px_rate + 1e-4).log() - (theta + 1e-4).log()
        global_recon = NegativeBinomial(theta, logits=nb_logits).sample()
        output_mean = NegativeBinomial(theta, logits=nb_logits).mean
        
        outputs = OutputsNB(z, global_recon, mu, logvar, output_mean)
            
        return outputs

class pmVAEModelNB(object):
    'A full model training wrapper for the pmVAE model'
    def __init__(self, 
        membership_mask,
        hidden_layers,
        latent_dim,
        cdim = None,
        hsic_penalty=None,
        activation='elu',
        batch_norm=True,
        bias_last_layer=False,
        add_auxiliary_module=False,
        use_gpu=True,
        **kwargs):
        '''
        Create a pmVAE for rna-seq.
        '''
        super(pmVAEModelNB, self).__init__()
        
        self.cdim = cdim
        self.hsic_penalty = hsic_penalty
        
        self.model = pmVAEnb( 
        membership_mask,
        hidden_layers,
        latent_dim,
        cdim=cdim,
        activation=activation,
        batch_norm=batch_norm,
        bias_last_layer=bias_last_layer,
        add_auxiliary_module=add_auxiliary_module,
        use_gpu=use_gpu,
        **kwargs)
        
        self.use_gpu=use_gpu
        
        if self.use_gpu:
            self.model.cuda()
        
    def calc_loss(self, data, val=False, use_c=False):
        
        if use_c:
            if self.use_gpu:
                x, c = data[0].float().cuda(), data[1].float().cuda()
            else:
                x, c = data[0].float(), data[1].long()
              
            x_ = torch.log(1.0 + x)
            network_input = torch.cat([x_, c], 1)
            
        else:
            if self.use_gpu:
                x = data.float().cuda()
            else:
                x = data.float()
            x_ = torch.log(1.0 + x)
            network_input = x_
            
        library = torch.sum(x, dim=1, keepdim=True)
        
        mu, logvar = self.model.encode(network_input)
        z = self.model.reparametrize(mu, logvar)
        generative_params = self.model.generative(z, library)
        px_rate = generative_params["px_rate"]
        theta = generative_params["theta"]
        qz_m = mu
        qz_v = torch.exp(logvar / 2)
        
        # term 1
        # the pytorch NB distribution uses a different parameterization
        # so we must apply a quick transformation (included in scvi-tools, but here we use the pytorch code)
        nb_logits = (px_rate + 1e-4).log() - (theta + 1e-4).log()
        log_lik = NegativeBinomial(theta, logits=nb_logits).log_prob(x).sum(dim=-1)
        
        if val:
            return -log_lik.mean()
        else:
            # term 2
            prior_dist = Normal(torch.zeros_like(qz_m), torch.ones_like(qz_v))
            var_post_dist = Normal(qz_m, torch.sqrt(qz_v))
            kl_divergence = kl(var_post_dist, prior_dist).sum(dim=1)
            elbo = log_lik - self.beta * kl_divergence
            loss = torch.mean(-elbo)
            return loss
        
    def get_recon_error(self, 
                        val_dataset,
                        batch_size=256):
        val_dataloader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=2)
        
        valloss = self._val_epoch(val_dataloader)
        valloss /= len(val_dataset)
        
        return valloss
        
        
    def train(self, 
              train_dataset, 
              val_dataset, 
              max_epochs=1200,
              lr=0.001,
              beta=1e-5,
#               weight_decay=1e-4,
              batch_size=256,
              pathway_dropout=True,
              logpath=None,
              checkpoint_path='pmvae_checkpoint.pkl',
              verbose=True):
        self.checkpoint_path = checkpoint_path
        self.lr = lr
        self.beta = beta
#         self.weight_decay = weight_decay
        
        # Create torch DataLoaders from the training and validation datasets.
        # Necessary for batching and shuffling data.
        train_dataloader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=2)
        val_dataloader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=2)
        
        self.x_dim = train_dataset.X.shape[1]
        
        self.optimizer = optim.Adam(self.model.parameters(),lr=self.lr)
        
        best_loss = None
        for i_epoch in range(max_epochs):
            print("-------- Epoch {:03d} --------".format(i_epoch))
            
            if self.cdim is not None:
                use_c=True
            else:
                use_c=False
                
            trainloss = self._train_epoch(train_dataloader, use_c=use_c)
            trainloss /= len(train_dataset)
            valloss = self._val_epoch(val_dataloader, use_c=use_c)
            valloss /= len(val_dataset)
            
            # only save if improvement
            if best_loss is None or valloss < best_loss: 
                best_loss = valloss
                self._checkpoint(i_epoch, valloss, suffix='.best_loss')
            else:
                self.lr = self.lr/10.
                self.optimizer = optim.Adam(self.model.parameters(),lr=self.lr)
            # Write information on this epoch to a log.
            logstr = "Epoch {:03d}: ".format(i_epoch) +\
                     "training loss {:08.4f},".format(trainloss) +\
                     "validation loss {:08.4f}".format(valloss)
            if not logpath is None:
                with open(logpath, 'a') as logfile:
                    logfile.write(logstr + '\n')
            if verbose:
                print(logstr)
        self.load_checkpoint(self.checkpoint_path+'.best_loss')
        
    def _train_epoch(self,train_dataloader,pathway_dropout=True,use_c=False):
        self.model.train()
        train_loss = 0
        for batch_idx, data in enumerate(train_dataloader):
            self.optimizer.zero_grad()
            loss = self.calc_loss(data, use_c=use_c)
            loss.backward()
            train_loss += loss.item()
            self.optimizer.step()
        return train_loss
    
    def _val_epoch(self,val_dataloader,use_c=False):
        self.model.train(False)
        val_loss = 0
        for batch_idx, data in enumerate(val_dataloader):
            loss = self.calc_loss(data, val=True, use_c=use_c)
            val_loss += loss.item()
        return val_loss
    
    def _checkpoint(self, epoch, valloss, suffix=None):
        '''
        Save a checkpoint to self.checkpoint_path, including the full model, 
        current epoch, learning rate, and random number generator state.
        '''
        state = {'model': self.model,
                 'best_loss': valloss,
                 'epoch': epoch,
                 'rng_state': torch.get_rng_state(),
                 'LR': self.lr ,
                 'optimizer': self.optimizer.state_dict()}
        checkpoint_path = self.checkpoint_path
        if suffix is not None:
            checkpoint_path = checkpoint_path + suffix
        torch.save(state, checkpoint_path)
    def load_checkpoint(self, path, load_optimizer=False):
        checkpoint = torch.load(path)
        self.model = checkpoint['model']
        if load_optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            
    def latent_space_names(self, terms=None):
        terms = self.model.terms if terms is None else terms
        assert terms is not None, 'Need to specify gene set terms'

        if self.model.add_auxiliary_module \
                and len(terms) == self.model.num_annotated_modules:
            terms = list(terms) + ['AUXILIARY']

        z = self.model._module_latent_dim
        repeated_terms = np.repeat(terms, z)
        index = np.tile(range(z), len(terms)).astype(str)
        latent_dim_names = map('-'.join, zip(repeated_terms, index))

        return list(latent_dim_names)
    
    def set_gpu(self, use_gpu):
        assert type(use_gpu) is bool, 'Argument must be "True" or "False"'
        if use_gpu:
            self.model.cuda()
            self.model.use_gpu = use_gpu
            self.use_gpu = use_gpu
            
        else:
            self.model.cpu()
            self.model.use_gpu = use_gpu
            self.use_gpu = use_gpu
            
