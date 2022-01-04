"""
"""

import pickle
import numpy as np
import torch
import json
from torch.utils.data import Dataset, DataLoader
from BayesianDTI.datahelper import *
from blitz.modules import BayesianLinear
from blitz.utils import variational_estimator
from abc import ABC, abstractmethod
import torch
import torch.distributions.studentT as studentT

##################################################################################################################
#
# Abstract classes for torch models.
#
##################################################################################################################

class AbstractMoleculeEncoder(ABC, torch.nn.Module):
    """
    Abstract base class of molecule embedding models.
    """
    
    def forward(self, src):
        emb = None
        return emb

    
class AbstractProteinEncoder(ABC, torch.nn.Module):
    """
    Abstract base class of protein embedding models.
    """
    
    def forward(self, src):
        emb = None
        return emb


class AbstractInteractionModel(ABC, torch.nn.Module):
    """
    Abstract base class of drug-target interaction models.
    """
    def forward(self, protein_emb, drug_emb):
        prediction = None
        return prediction
    
    
class AbstractDTIModel(ABC, torch.nn.Module):
    def __init__(self):
        super(AbstractDTIModel, self).__init__()
        self.protein_encoder = AbstractMoleculeEncoder()
        self.smiles_encoder = AbstractProteinEncoder()
        self.interaction_predictor = AbstractInteractionModel()
        
    def forward(self, d, p):
        """
        Args:
            d(Tensor) : Preprocessed drug input batch
            p(Tensor) : Preprocessed protein input batch
            
            both d and p contains Long elements representing the token,
            such as
            ["C", "C", "O", "H"] -> Tensor([4, 4, 5, 7])
            ["P, K"] -> Tensor([12, 8])
            
        Return:
            (Tensor) [batch_size, 1]: predicted affinity value
        """
        p_emb = self.protein_encoder(p)
        d_emb = self.smiles_encoder(d)
        
        return self.interaction_predictor(p_emb, d_emb)
    
##################################################################################################
##################################################################################################
##################################################################################################

class MoleculeTransformer(AbstractMoleculeEncoder):
    """
    GPU memory consumption: 5939 Mb
    
    """
    def __init__(self, ntoken, ninp=128, nhead=8, nhid=512, nlayers=8, dropout=0.1):
        super(MoleculeTransformer, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.ninp = ninp

        ###################################################################
        self.pos_encoder = torch.nn.Embedding(100, ninp)
        self.encoder = torch.nn.Embedding(ntoken, ninp)

        ###################################################################
        self.layer_norm = torch.nn.LayerNorm([ninp])
        self.output_layer_norm = torch.nn.LayerNorm([ntoken])
        self.input_layer_norm = torch.nn.LayerNorm([ninp])

        encoder_layers = TransformerEncoderLayer(d_model=ninp,
                                                 nhead=nhead,
                                                 dim_feedforward=nhid,
                                                 dropout=dropout,
                                                 activation='gelu')

        self.transformer_encoder = TransformerEncoder(encoder_layers,
                                                      nlayers,
                                                      norm=self.layer_norm)

        self.dropout = torch.nn.Dropout(dropout)
        ###################################################################
        self.decoder = torch.nn.Linear(ninp, ntoken, bias=False) ## embedded -> seq
        self.decoder_bias = torch.nn.Parameter(torch.zeros(ntoken))
        self.init_weights()


    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.normal_(mean=0.0, std=1.0)
        self.decoder.weight.data.normal_(mean=0.0, std=1.0)
        self.decoder_bias.data.zero_()

        self.input_layer_norm.weight.data.fill_(1.0)
        self.input_layer_norm.bias.data.zero_()
        self.output_layer_norm.weight.data.fill_(1.0)
        self.output_layer_norm.bias.data.zero_()
        self.layer_norm.weight.data.fill_(1.0)
        self.layer_norm.bias.data.zero_()


    def forward(self, src, latent_out=False):
        """
        latent_out:
            If latent_out == true, the model will return the transformer encoding latent values,
            not a decoded vectors using self.decoder.
        """
        pos = torch.arange(0,100).long().to(src.device)

        mol_token_emb = self.encoder(src)
        pos_emb = self.pos_encoder(pos) ### Input embedding = positional embedding + normal embedding
        input_emb = pos_emb + mol_token_emb
        input_emb = self.input_layer_norm(input_emb) ## Should we use this?
        input_emb = self.dropout(input_emb)
        input_emb = input_emb.transpose(0, 1) ## Should we transpose this??..

        attention_mask = torch.ones_like(src).to(src.device)
        attention_mask = attention_mask.masked_fill(src!=1., 0.)
        attention_mask = attention_mask.bool().to(src.device)

        output = self.transformer_encoder(input_emb)#, src_key_padding_mask=attention_mask) ### Self-attention layers : dim = ninp
        
        if latent_out:
            return output
        output = self.decoder(output) + self.decoder_bias ### decoding

        return output


class SMILESEncoder(AbstractMoleculeEncoder):
    def __init__(self, smile_len=64+1, latent_len=128): ## +1 for 0 padding
        super(SMILESEncoder, self).__init__()
        self.encoder = torch.nn.Embedding(smile_len, latent_len)
        self.conv1 = torch.nn.Conv1d(latent_len, 32, 4)
        self.conv2 = torch.nn.Conv1d(32, 64, 6)
        self.conv3 = torch.nn.Conv1d(64, 96, 8)
        
        torch.nn.init.kaiming_normal_(self.encoder.weight)
        torch.nn.init.kaiming_normal_(self.conv1.weight)
        torch.nn.init.kaiming_normal_(self.conv2.weight)
        torch.nn.init.kaiming_normal_(self.conv3.weight)
        torch.nn.init.kaiming_normal_(self.encoder.weight)
        
    def forward(self, src):
        emb = self.encoder(src)
        conv1 = torch.nn.ReLU()(self.conv1(emb.transpose(1,2)))
        conv2 = torch.nn.ReLU()(self.conv2(conv1))
        conv3 = torch.nn.ReLU()(self.conv3(conv2))
        
        return torch.max(conv3, 2)[0]
        
        
class ProteinEncoder(AbstractProteinEncoder):
    def __init__(self, protein_len=25+1, latent_len=128): ## +1 for 0 padding
        super(ProteinEncoder, self).__init__()
        self.encoder = torch.nn.Embedding(protein_len, latent_len)
        self.conv1 = torch.nn.Conv1d(latent_len, 32, 4)
        self.conv2 = torch.nn.Conv1d(32, 64, 8)
        self.conv3 = torch.nn.Conv1d(64, 96, 12)
        
        torch.nn.init.kaiming_normal_(self.encoder.weight)
        torch.nn.init.kaiming_normal_(self.conv1.weight)
        torch.nn.init.kaiming_normal_(self.conv2.weight)
        torch.nn.init.kaiming_normal_(self.conv3.weight)
        torch.nn.init.kaiming_normal_(self.encoder.weight)
    
    def forward(self, src):
        emb = self.encoder(src)
        conv1 = torch.nn.ReLU()(self.conv1(emb.transpose(1,2)))
        conv2 = torch.nn.ReLU()(self.conv2(conv1))
        conv3 = torch.nn.ReLU()(self.conv3(conv2))
        
        return torch.max(conv3, 2)[0]
    
    
class InteractionPredictor(AbstractInteractionModel):
    def __init__(self, input_dim):
        super(InteractionPredictor, self).__init__()
        self.fully1 = torch.nn.Linear(input_dim, 1024)
        self.fully2 = torch.nn.Linear(1024, 1024)
        self.fully3 = torch.nn.Linear(1024, 512)
        self.output = torch.nn.Linear(512, 1)
        self.dropout = torch.nn.Dropout(0.1)
    
    
    def forward(self, protein_emb, drug_emb):
        src = torch.cat((protein_emb, drug_emb), 1)
        fully1 = torch.nn.ReLU()(self.fully1(src))
        fully1 = self.dropout(fully1)
        fully2 = torch.nn.ReLU()(self.fully2(fully1))
        fully2 = self.dropout(fully2)
        fully3 = torch.nn.ReLU()(self.fully3(fully2))
        return self.output(fully3)
    
    
@variational_estimator
class InteractionEpistemicUncertaintyPredictor(AbstractInteractionModel):
    """
    Interaction layers to estimate Epistemic uncertainty using Bayesian fully conneceted layers
    using Bayes By Backprop(with blitz library).
    
    """
    def __init__(self, input_dim):
        super(InteractionEpistemicUncertaintyPredictor, self).__init__()
        self.fully1 = torch.nn.Linear(input_dim, 1024)#, prior_sigma_1=0.1, prior_sigma_2 = 0.0002)
        self.fully2 = torch.nn.Linear(1024, 1024)#, prior_sigma_1 = 0.1, prior_sigma_2 = 0.0002)
        self.fully3 = BayesianLinear(1024, 512, prior_sigma_1 = 0.5)
        self.output = BayesianLinear(512, 1, prior_sigma_1 = 0.5)
        
    
    def forward(self, protein_emb, drug_emb):
        src = torch.cat((protein_emb, drug_emb), 1)
        fully1 = torch.nn.ReLU()(self.fully1(src))
        fully2 = torch.nn.ReLU()(self.fully2(fully1))
        fully3 = torch.nn.ReLU()(self.fully3(fully2))
    
        return self.output(fully3)

    
class InteractionAleatoricUncertaintyPredictor(InteractionPredictor):
    """
    Interaction layers to estimate Aleatoric uncertainty using the unimodel Gaussian
    output.
    
    The final variance can be calculated as follows:
        var = log(exp( 1 + X ))
    """
    def __init__(self, input_dim):
        super(InteractionAleatoricUncertaintyPredictor, self).__init__(input_dim)
        self.output = torch.nn.Linear(512, 1)
        self.output_uncertainty = torch.nn.Linear(512, 1)
        torch.nn.init.kaiming_normal_(self.output.weight)
        
    def forward(self, protein_emb, drug_emb):
        src = torch.cat((protein_emb, drug_emb), 1)
        fully1 = torch.nn.ReLU()(self.fully1(src))
        fully1 = self.dropout(fully1)
        fully2 = torch.nn.ReLU()(self.fully2(fully1))
        fully2 = self.dropout(fully2)
        fully3 = torch.nn.ReLU()(self.fully3(fully2))
        
        mean = self.output(fully3)
        var = torch.log(torch.exp(self.output_uncertainty(fully3)) + 1)
        
        return mean, var


class EvidentialLinear(torch.nn.Module):
    """
    *Note* The layer should be putted at the final of the model architecture.
    
    The output of EvidentialLineary layer is parameters of Normal-Inverse-Gamma (NIG) distribution.
    We can generate the probability distribution of the target value by using the output of this layer.
    
    The inverse-normal-gamma distribution can be formulated:
    y ~ Normal(mu, sigma**2)
    mu ~ Normal(gamma, (T/nu)**2)
    sigma ~ InverseGamma(alpha, beta)
    
    where y is a target value such as a durg-target affinity value.

    However, when we train the Evidential network and predict target values by the model,
    we do not directly use the NIG distribution. Our output probability distribution is the distribution
    by analytically marginalizing out mu and sigma[(https://arxiv.org/pdf/1910.02600); equation 6, 7].

    ************************************************************************************************
    *** Target probability distribution:
    *** p(y|gamma, nu, alpha, beta) = t-distribution(gamma, beta*(1+nu)/(nu*alpha) , 2*alpha)
    *** 
    *** We can train and infer the true value "y" by using the above probability distribution.
    ************************************************************************************************

    Args:
        gamma(Tensor): The parameter of the NIG distribution. This is the predictive value (predictive mean)
            of the output distribution.
        nu(Tensor): The parameter of the NIG distribution.
        alpha(Tensor): The parameter of the NIG distribution.
        beta(Tensor): The parameter of the NIG distribution.
    """
    def __init__(self, input_dim, output_dim=1):
        """[summary]

        Args:
            input_dim ([type]): [description]
            output_dim (int, optional): [description]. Defaults to 1.
        """
        self.gamma = torch.nn.Linear(input_dim, output_dim)
        self.nu = torch.nn.Linear(input_dim, output_dim)
        self.alpha = torch.nn.Linear(input_dim, output_dim)
        self.beta = torch.nn.Linear(input_dim, output_dim)

    def forward(self, src):
        gamma = self.gamma(src)
        alpha = torch.nn.Softplus()(self.alpha(src)) + 1
        beta = torch.nn.Softplus()(self.beta(src))
        nu = torch.nn.Softplus()(self.nu(src))

        return gamma, alpha, beta, nu


class InteractionEvidentialNetwork(AbstractInteractionModel):
    """
    Deep evidential regression - (https://arxiv.org/pdf/1910.02600)
    
    Interaction layers using Deep-evidential regression. The output neurons of this network
    are the parameter of Inverse-Normal-Gamma distribution, which is the conjugate prior of Normal.
    
    The inverse-normal-gamma distribution can be formulated:
    X ~ N(gamma, T/nu)
    T ~ InverseGamma(alpha, beta)
    
    So we can make t-distribution distribution using the parameters {gamma, nu, alpha, beta}, which are
    the output of this network.
    """
    def __init__(self, input_dim):
        super(InteractionEvidentialNetwork, self).__init__()
        self.fully1 = torch.nn.Linear(input_dim, 1024)
        self.fully2 = torch.nn.Linear(1024, 1024)
        self.fully3 = torch.nn.Linear(1024, 512)
        
        self.gamma = torch.nn.Linear(512, 1)
        self.nu = torch.nn.Linear(512, 1)#, bias=False)
        self.alpha = torch.nn.Linear(512, 1)#, bias=False)
        self.beta = torch.nn.Linear(512, 1)#, bias=False)

    def forward(self, protein_emb, drug_emb):
        src = torch.cat((protein_emb, drug_emb), 1)
        fully1 = torch.nn.ReLU()(self.fully1(src))
        fully2 = torch.nn.ReLU()(self.fully2(fully1))
        fully3 = torch.nn.ReLU()(self.fully3(fully2))
        
        gamma = self.gamma(fully3)
        alpha = torch.nn.Softplus()(self.alpha(fully3)) + 1
        beta = torch.nn.Softplus()(self.beta(fully3))
        nu = torch.nn.Softplus()(self.nu(fully3))
        
        return gamma, nu, alpha, beta

    
class InteractionEvidentialNetworkDropout(InteractionEvidentialNetwork):
    """[summary]

    Args:
        InteractionEvidentialNetwork ([type]): [description]
    """
    def __init__(self, input_dim):
        super(InteractionEvidentialNetworkDropout, self).__init__(input_dim)
        self.dropout = torch.nn.Dropout(0.1)
        
    def forward(self, protein_emb, drug_emb):
        src = torch.cat((protein_emb, drug_emb), 1)
        fully1 = self.dropout(torch.nn.ReLU()(self.fully1(src)))
        fully2 = self.dropout(torch.nn.ReLU()(self.fully2(fully1)))
        fully3 = torch.nn.ReLU()(self.fully3(fully2))
        
        gamma = self.gamma(fully3)
        alpha = torch.nn.Softplus()(self.alpha(fully3)) + 1
        beta = torch.nn.Softplus()(self.beta(fully3))
        nu = torch.nn.Softplus()(self.nu(fully3))
        
        return gamma, nu, alpha, beta


class InteractionEvidentialNetworkMTL(InteractionEvidentialNetwork):
    """
    Multi-task version of the evidential DTI network.
    The fully-connected modules of the network are divided, one is the
    point-estimation network(fully3) and the other is the bayesian-inference network(fully3_var)

    Methods:
        __init__()

        forward()
    """    
    def __init__(self, input_dim):
        super(InteractionEvidentialNetworkMTL, self).__init__(input_dim)
        self.fully3_var = torch.nn.Linear(1024, 512)
        #self.fully2_var = torch.nn.Linear(1024, 1024)
        self.dropout = torch.nn.Dropout(0.1)
        
    def forward(self, protein_emb, drug_emb):
        src = torch.cat((protein_emb, drug_emb), 1)
        fully1 = self.dropout(torch.nn.ReLU()(self.fully1(src)))
        fully2 = self.dropout(torch.nn.ReLU()(self.fully2(fully1)))
        fully3 = self.dropout(torch.nn.ReLU()(self.fully3(fully2)))
        #fully2_var = self.dropout(torch.nn.ReLU()(self.fully2_var(fully1)))
        fully3_var = self.dropout(torch.nn.ReLU()(self.fully3_var(fully2)))
        
        gamma = self.gamma(fully3)
        alpha = torch.nn.Softplus()(self.alpha(fully3_var)) + 1
        beta = torch.nn.Softplus()(self.beta(fully3_var))
        nu = torch.nn.Softplus()(self.nu(fully3_var))
         
        return gamma, nu, alpha, beta
    
        
        
@variational_estimator
class BayesianInteractionPredictor(AbstractInteractionModel):
    """
    Interaction layers to estimate Epistemic uncertainty using Bayesian fully conneceted layers
    using Bayes By Backprop(with blitz library).
    
    """
    def __init__(self, input_dim):
        super(BayesianInteractionPredictor, self).__init__()
        self.fully1 = BayesianLinear(input_dim, 1024)#, prior_sigma_1=0.1, prior_sigma_2 = 0.0002)
        self.fully2 = BayesianLinear(1024, 1024)#, prior_sigma_1 = 0.1, prior_sigma_2 = 0.0002)
        self.fully3 = BayesianLinear(1024, 512)#, prior_sigma_1 = 0.1, prior_sigma_2 = 0.0002)
        self.output = BayesianLinear(512, 1)#, prior_sigma_1 = 0.1, prior_sigma_2 = 0.0002)
        
    
    def forward(self, protein_emb, drug_emb):
        src = torch.cat((protein_emb, drug_emb), 1)
        fully1 = torch.nn.ReLU()(self.fully1(src))
        fully2 = torch.nn.ReLU()(self.fully2(fully1))
        fully3 = torch.nn.ReLU()(self.fully3(fully2))
    
        return self.output(fully3)
    

class DeepDTA(AbstractDTIModel):
    """
    The final DeepDTA model includes the protein encoding model;
    the smiles(drug; chemical) encoding model; the interaction model.
    """
    def __init__(self, concat_dim=96*2):
        super(DeepDTA, self).__init__()
        self.protein_encoder = ProteinEncoder()
        self.smiles_encoder = SMILESEncoder()
        self.interaction_predictor = InteractionPredictor(concat_dim)
        
    def forward(self, d, p):
        """
        Args:
            d(Tensor) : Preprocessed drug input batch
            p(Tensor) : Preprocessed protein input batch
            
            both d and p contains Long elements representing the token,
            such as
            ["C", "C", "O", "H"] -> Tensor([4, 4, 5, 7])
            ["P, K"] -> Tensor([12, 8])
            
        Return:
            (Tensor) [batch_size, 1]: predicted affinity value
        """
        p_emb = self.protein_encoder(p)
        d_emb = self.smiles_encoder(d)
        
        return self.interaction_predictor(p_emb, d_emb)
    
    def train_dropout(self):
        def turn_on_dropout(m):
            if type(m) == torch.nn.modules.dropout.Dropout:
                m.train()
        self.apply(turn_on_dropout)
    
    
class DeepDTAAleatoricBayes(AbstractDTIModel):
    """
    DeepDTA model with Aleatoric uncertainty modeling using the
    unimodel Gaussian output, which can model the noise(uncertaitny) of data itself.
    """
    def __init__(self, concat_dim=96*2):
        super(DeepDTAAleatoricBayes, self).__init__()
        self.protein_encoder = ProteinEncoder()
        self.smiles_encoder = SMILESEncoder()
        self.interaction_predictor = InteractionAleatoricUncertaintyPredictor(concat_dim)
        
    def forward(self, d, p):
        """
        Args:
            d(Tensor) : Preprocessed drug input batch
            p(Tensor) : Preprocessed protein input batch
        """
        p_emb = self.protein_encoder(p)
        d_emb = self.smiles_encoder(d)
        
        return self.interaction_predictor(p_emb, d_emb)
    

@variational_estimator
class DeepDTAEpistemicBayes(AbstractDTIModel):
    """
    DeepDTA model with Epistemic uncertainty modeling using the
    Bayesian neural network layer, which can model the uncertainty of model parameters(posteriors).
    """
    def __init__(self, concat_dim=96*2):
        super(DeepDTAEpistemicBayes, self).__init__()
        self.protein_encoder = ProteinEncoder()
        self.smiles_encoder = SMILESEncoder()
        self.interaction_predictor = BayesianInteractionPredictor(concat_dim)
        
    def forward(self, input_, sample_nbr=10):
        """
        Args:
            input_(tuple) -> (d, p)
                d(Tensor) : Preprocessed drug input batch
                p(Tensor) : Preprocessed protein input batch
        """
        p_emb = self.protein_encoder(input_[1])
        d_emb = self.smiles_encoder(input_[0])
        
        return self.interaction_predictor(p_emb, d_emb)
    
    def mfvi_forward_dti(self, d, p, sample_nbr=10):
        """
        Overroaded original method from blitz library
        
        Performs mean-field variational inference for the variational estimator model:
            Performs sample_nbr forward passes with uncertainty on the weights, returning its mean and standard deviation
        Parameters:
            inputs: torch.tensor -> the input data to the model
            sample_nbr: int -> number of forward passes to be done on the data
        Returns:
            mean_: torch.tensor -> mean of the perdictions along each of the features of each datapoint on the batch axis
            std_: torch.tensor -> std of the predictions along each of the features of each datapoint on the batch axis
        """
        result = torch.stack([self(d, p) for _ in range(sample_nbr)])
        return result.mean(dim=0), result.std(dim=0)

    
@variational_estimator
class DeepDTAEpistemicBayesStudent(DeepDTAEpistemicBayes):
    """
    DeepDTA model with Epistemic uncertainty modeling using the
    Bayesian neural network layer, which can model the uncertainty of model parameters(posteriors).
    """
    def __init__(self, concat_dim=96*2):
        super(DeepDTAEpistemicBayesStudent, self).__init__(concat_dim=concat_dim)
        self.interaction_predictor = InteractionEpistemicUncertaintyPredictor(concat_dim)
        
    def forward(self, d, p, sample_nbr=10):
        """
        If sample_nbr == 1, it is just one-time prediction with weight sampling.
        
        Args:
            d(Tensor) : Preprocessed drug input batch
            p(Tensor) : Preprocessed protein input batch
        
        """
        p_emb = self.protein_encoder(p)
        d_emb = self.smiles_encoder(d)
        
        predictions = torch.stack([self.interaction_predictor(p_emb, d_emb) for _ in range(sample_nbr)]) # S*N*1
        mean = predictions.mean(dim=0) # N*1
        var = torch.sum((predictions - mean)**2, dim=0) / (sample_nbr - 1)
        std = var**0.5
        
        # ISSUE: prediction.std(dim=0) raise a error.
        # prediction.mean(dim=0) did not raise a error
        return mean, std#torch.std(predictions, dim=0)


class EvidentialDeepDTA(DeepDTA):
    """
    DeepDTA model with Prior interaction networks.
    
    """
    def __init__(self, concat_dim=96*2, dropout=True, mtl=False):
        super(EvidentialDeepDTA, self).__init__(concat_dim=concat_dim)
        if dropout:
            self.interaction_predictor = InteractionEvidentialNetworkDropout(concat_dim)
            if mtl:
                self.interaction_predictor = InteractionEvidentialNetworkMTL(concat_dim)
        else:
            self.interaction_predictor = InteractionEvidentialNetwork(concat_dim)
    
    def forward(self, d, p):
        output_tensors = super().forward(d, p)
            
        return output_tensors
    
    @staticmethod
    def aleatoric_uncertainty(nu, alpha, beta):
        return torch.sqrt(beta/(alpha-1))
    
    @staticmethod
    def epistemic_uncertainty(alpha, beta):
        return torch.sqrt(beta/(nu*(alpha-1)))
    
    @staticmethod
    def total_uncertainty(nu,alpha,beta):
        """
        Return standard deviation of Generated student t distribution,
        
        p(y|gamma, nu, alpha, beta)  = Student-t(y; gamma, beta*(1+nu)/(nu*alpha), 2*alpha )
        
        Note that the freedom of given student-t distribution is 2alpha.
        """
        return torch.sqrt(beta*(1+nu)/(nu*alpha))

    def predictive_entropy(self, nu, alpha, beta):
        scale = (beta*(1+nu)/(nu*alpha)).sqrt()
        df = 2*alpha
        dist = studentT.StudentT(df = df, scale=scale)
        return dist.entropy()

    @staticmethod
    def freedom(alpha):
        return 2*alpha
    
class DeepDTAPriorNetwork(EvidentialDeepDTA):
    """
    Class to load old models
    Args:
        EvidentialDeepDTA ([type]): [description]
    """
    def __init__(self, concat_dim=96*2, dropout=True, mtl=False):
        super(DeepDTAPriorNetwork, self).__init__(concat_dim=concat_dim, dropout=dropout, mtl=mtl)
    def forward(self, d, p):
        return super(DeepDTAPriorNetwork, self).forward(d, p)

class InteractionPriorNetworkDropout(InteractionEvidentialNetwork):
    def __init__(self, input_dim):
        super(InteractionPriorNetworkDropout, self).__init__(input_dim)
    
    def forward(self, protein_emb, drug_emb):
        return super(InteractionPriorNetworkDropout, self).forward(protein_emb, drug_emb)

# Experimental model
class uncertaintyEstimationNetwork(torch.nn.Module):
    """
    ***************************
    ****** Experimental *******
    ***************************
    
    Maybe seperated uncertainty estimator could measure true uncertainty? 
    
    """
    def __init__(self, concat_dim=96*2):
        super(InteractionPredictor, self).__init__()
        
