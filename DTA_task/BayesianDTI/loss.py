"""
Custom Loss function for Bayesian approximations
"""
import torch
import numpy as np
from BayesianDTI.datahelper import *
from BayesianDTI.utils import get_mse_coef_test

pi = 3.141592741012573


class GaussianNLL(torch.nn.modules.loss._Loss):
    """
    Negative Gaussian likelihood loss.
    """
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean'):
        super(GaussianNLL, self).__init__(size_average, reduce, reduction)
    
    def forward(self, input_mu: torch.Tensor, input_std: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        x1 = 0.5*torch.log(2*pi*input_std*input_std)
        x2 = 0.5/(input_std**2)*((target - input_mu)**2)
        
        if self.reduction == 'mean':
            return torch.mean(x1 + x2)
        elif self.reduction == 'sum':
            return torch.sum(x1 + x2)


class MarginalLikelihoodLoss(torch.nn.modules.loss._Loss):
    """
    Marginal likelihood error of prior network.
    The target value is not a distribution (mu, std), but a just value.
    
    This is a negative log marginal likelihood, with marginalized mu and sigma.
    
    """
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean', eps=1e-9):
        super(MarginalLikelihoodLoss, self).__init__(size_average, reduce, reduction)
    
    def forward(self, gamma: torch.Tensor, nu: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            gamma, nu, alpha, beta -> outputs of prior network
            
            target -> target value
            
        Return:
            (Tensor) Negative log marginal likelihood of PriorNet
                p(y|m) = Student-t(y; gamma, (beta(1+nu))/(nu*alpha) , 2*alpha)

                then, the negative log likelihood is (CAUTION QUITE COMPLEX!)

                NLL = -log(p(y|m)) =
                    log(3.14/nu)*0.5 - alpha*log(2*beta*(1 + nu)) + (alpha + 0.5)*log( nu(target - gamma)^2 + 2*beta(1 + nu) )
                    + log(GammaFunc(alpha)/GammaFunc(alpha + 0.5))
        """
        pi = torch.tensor(np.pi)
        
        x1 = torch.log(pi/nu)*0.5
        x2 = -alpha*torch.log(2.*beta*(1.+ nu))
        x3 = (alpha + 0.5)*torch.log( nu*(target - gamma)**2 + 2.*beta*(1. + nu) )
        x4 = torch.lgamma(alpha) - torch.lgamma(alpha + 0.5)
        
        loss = x1 + x2 + x3 + x4
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss 


class ModifiedMSE(torch.nn.modules.loss._Loss):
    """
    Modified MSE loss of the MT-ENet

    Returns:
        (Tensor) Modified MSE loss value.
    """
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean'):
        super(ModifiedMSE, self).__init__(size_average, reduce, reduction)

    def forward(self, gamma: torch.Tensor, nu: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        mse = (gamma-target)**2
        c = get_mse_coef_test(gamma, nu, alpha, beta, target).detach()
        modified_mse = mse*c
        if self.reduction == 'mean':
            return modified_mse.mean()
        elif self.reduction == 'sum':
            return modified_mse.sum()
        else:
            return modified_mse

def modified_mse(gamma, nu, alpha, beta, target, **kwargs):
    mse = (gamma-target)**2
    c = get_mse_coef_test(gamma, nu, alpha, beta, target, **kwargs).detach()
    modified_mse = mse*c
    return modified_mse.mean()

class EvidenceRegularizer(torch.nn.modules.loss._Loss):
    """
    Regularization for the regression prior network.
    If the self.factor increases, the model output the wider(high confidence interval) predictions.
    """
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean', eps=1e-9, factor=0.1):
        super(EvidenceRegularizer, self).__init__(size_average, reduce, reduction)
        self.factor = factor
    
    def forward(self, gamma: torch.Tensor, nu: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            gamma, nu, alpha, beta -> outputs of prior network
            
            target -> target value
            
            factor -> regularization strength
        
        Return:
            (Tensor) prior network regularization
            Loss = |y - gamma|*(2*nu + alpha) * factor
            
        """
        loss = torch.abs(target - gamma)*(2*nu + alpha) * self.factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss 

    
class JensenShannonDivergenceLoss(torch.nn.modules.loss._Loss):
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean', eps=1e-9):
        super(JensenShannonDivergenceLoss, self).__init__(size_average, reduce, reduction)
        self.eps = eps
    def forward(self, input_mu: torch.Tensor, input_std: torch.Tensor,
                target_mu: torch.Tensor, target_std: torch.Tensor) -> torch.Tensor:
        pass
    
    
class GaussianKLDLoss(torch.nn.modules.loss._Loss):
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean', eps=1e-9):
        super(GaussianKLDLoss, self).__init__(size_average, reduce, reduction)
        self.eps = eps
        
    def forward(self, input_mu: torch.Tensor, input_std: torch.Tensor,
                target_mu: torch.Tensor, target_std: torch.Tensor) -> torch.Tensor:
        """
        Calculate KL-divergence between two univariate gaussian point-wisely.
        Somehow, this is numerically highly unstable although it added the eps for variances.
        
        Predicted gaussian ~ N(input_mu, input_std)
        Target gaussian ~ N(target_mu, input_std)
        
        Args:
            input_mu(Tensor)
            input_std(Tensor)
            target_mu(Tensor)
            target_std(Tensor)
            
        Return:
            (Tensor) KLD Loss.
        """
        target_std += self.eps
        input_std += self.eps
        x1 = torch.log(target_std/input_std)
        x2 = (input_std**2 + (input_mu - target_mu)**2) / (2*(target_std**2))
        
        if self.reduction == 'sum':
            return torch.sum(x1 + x2 - 0.5)
        elif self.reduction == 'mean':
            return torch.mean(x1 + x2 - 0.5)
        
    def sampling_forward(self, predict_samples, target_mu, target_std):
        pass


class DoubleGaussianKLDLoss(GaussianKLDLoss):
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean', eps=1e-9):
        super(DoubleGaussianKLDLoss, self).__init__(size_average, reduce, reduction)
        self.eps = eps
    
    def forward(self, input_mu: torch.Tensor, input_std: torch.Tensor,
               target_mu: torch.Tensor, target_std: torch.Tensor) -> torch.Tensor:
        """
        Calculate reverse and forward KL-divergence, and take average.
        """
        
        target_std += self.eps
        input_std += self.eps
        x1 = torch.log(target_std/input_std)
        x2 = (input_std**2 + (input_mu - target_mu)**2) / (2*(target_std**2))
        forward = x1 + x2 - 0.5
        
        x1 = torch.log(input_std/target_std)
        x2 = (target_std**2 + (target_mu - input_mu)**2) / (2*(input_std**2))
        reverse = x1 + x2 - 0.5
        
        if self.reduction == 'mean':
            return torch.mean((forward + reverse)/2)
        elif self.reduction == 'sum':
            return torch.sum((forward + reverse)/2)
    

class DoubleMSELoss(torch.nn.modules.loss._Loss):
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean', alpha=1.0):
        super(DoubleMSELoss, self).__init__(size_average, reduce, reduction)
        self.alpha = 1.0
    
    def forward(self, input_mu: torch.Tensor, input_std: torch.Tensor,
                target_mu: torch.Tensor, target_std: torch.Tensor) -> torch.Tensor:
        """
        Calculate MSE for both means and stds.
        
        """
        loss = (input_mu - target_mu)**2 + self.alpha*(input_std - target_std)**2
        
        if self.reduction == 'sum':
            return torch.sum(loss)
        elif self.reduction == 'mean':
            return torch.mean(loss)

class Wasserstein2Loss(torch.nn.modules.loss._Loss):
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean', scale=1.0):
        super(Wasserstein2Loss, self).__init__(size_average, reduce, reduction)
        self.scale = scale
        
    def forward(self, input_mu: torch.Tensor, input_std: torch.Tensor,
                target_mu: torch.Tensor, target_std: torch.Tensor) -> torch.Tensor:
        """
        Calculate 1`th wasserstein distance between two univariate gaussian point-wisely.
        
        Predicted gaussian ~ N(input_mu, input_std)
        Target gaussian ~ N(target_mu, input_std)
        
        wasserstein 2 between two univarate gaussian can be caluclated by following the rules
        
        W^2(N(m1, s1), N(m2, s2))
        
        Args:
            input_mu(Tensor)
            input_std(Tensor)
            target_mu(Tensor)
            target_std(Tensor)
            
        Return:
            (Tensor) Wasserstein Loss.
        
        https://stats.stackexchange.com/questions/83741/earth-movers-distance-emd-between-two-gaussians
        
        """
        x_1 = (input_mu - target_mu)**2
        x_2 = input_std + target_std - 2*torch.sqrt(input_std*target_std)
        
        if self.reduction == 'sum':
            return torch.sum(x_1 + x_2)*self.scale
        
        elif self.reduction == 'mean':
            if torch.isnan(torch.mean(x_1+x_2)):
                print(x_1, x_2)
                print(input_std, target_std)
                assert False, "Why Nan happen?"
            return torch.mean(x_1 + x_2)*self.scale
        else:
            return (x_1 + x_2)*self.scale
        
        
class Wasserstein1Loss(torch.nn.modules.loss._Loss):
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean', scale=1.0):
        super(Wasserstein1Loss, self).__init__(size_average, reduce, reduction)
        self.scale = scale
        
    def forward(self, input_mu: torch.Tensor, input_std: torch.Tensor,
                target_mu: torch.Tensor, target_std: torch.Tensor) -> torch.Tensor:
        """
        Calculate 2`ed wasserstein distance between two univariate gaussian point-wisely.
        
        Predicted gaussian ~ N(input_mu, input_std)
        Target gaussian ~ N(target_mu, input_std)
        
        wasserstein between two univarate gaussian can be caluclated by following the rules
        
        W(N(m1, s1), N(m2, s2))
        
        Args:
            input_mu(Tensor)
            input_std(Tensor)
            target_mu(Tensor)
            target_std(Tensor)
            
        Return:
            (Tensor) Wasserstein Loss.
        
        https://stats.stackexchange.com/questions/83741/earth-movers-distance-emd-between-two-gaussians
        
        """
        x_1 = torch.abs(input_mu - target_mu)
        x_2 = torch.sqrt( (input_std**0.5 - target_std**0.5)**2
                         + 2*((target_std*input_std)**0.5)*(1 - target_std*input_std) )
        
        if self.reduction == 'sum':
            return torch.sum(x_1 + x_2)*self.scale
        
        elif self.reduction == 'mean':
            if torch.isnan(torch.mean(x_1+x_2)):
                print(x_1, x_2)
                print(input_std, target_std)
                assert False, "Why Nan happen?"
            return torch.mean(x_1 + x_2)*self.scale
        
        
class IndividualKLDLoss(torch.nn.modules.loss._Loss):
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean'):
        super(IndividualKLDLoss, self).__init__(size_average, reduce, reduction)
        
    def forward(self, input_mu: torch.Tensor, input_std: torch.Tensor,
                target_mu: torch.Tensor, target_std: torch.Tensor) -> torch.Tensor:
        pass
    

class LaplacePriorKLD(torch.nn.modules.loss._Loss):
    """
    NIPS2020 tutorial: "Bayesian neural networks priors revisited" suggests
    that Laplace prior is a much more well-fitted distribution for BNN, since the
    empirical distribution of neural network seems like heavy-tailed(Laplace).
    """
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean'):
        super(LaplacePriorKLD, self).__init__(size_average, reduce, reduction)
        
    def forward(self, input_mu: torch.Tensor, input_std: torch.Tensor) -> torch.Tensor:
        pass