import torch
import numpy as np
from utils import *

class EvidentialNetwork(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=100):
        super(EvidentialNetwork, self).__init__()
        self.fully1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fully2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fully3 = torch.nn.Linear(hidden_dim, hidden_dim)
        
        self.gamma = torch.nn.Linear(hidden_dim, 1)
        self.nu = torch.nn.Linear(hidden_dim, 1)
        self.alpha = torch.nn.Linear(hidden_dim, 1)
        self.beta = torch.nn.Linear(hidden_dim, 1)
        self.dropout = torch.nn.Dropout(0.0)

    def forward(self, src):
        fully1 = torch.nn.Tanh()(self.fully1(src))
        fully2 = torch.nn.Tanh()(self.fully2(fully1))
        fully3 = torch.nn.Tanh()(self.fully3(fully2))
        
        gamma = self.gamma(fully3)
        alpha = torch.nn.Softplus()(self.alpha(fully3)) + 1 #+ 1E-9
        beta = torch.nn.Softplus()(self.beta(fully3))
        nu = torch.nn.Softplus()(self.nu(fully3))# + 1E-9
        
        gamma.retain_grad()
        nu.retain_grad()
        alpha.retain_grad()
        beta.retain_grad()
        
        return gamma, nu, alpha, beta

    def last_latent(self,src):
        fully1 = self.dropout(torch.nn.Tanh()(self.fully1(src)))
        fully2 = self.dropout(torch.nn.Tanh()(self.fully2(fully1)))
        fully3 = self.dropout(torch.nn.Tanh()(self.fully3(fully2)))
        return fully3

class EvidentialNetwork2(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=100):
        super(EvidentialNetwork2, self).__init__()
        self.fully1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fully2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fully3 = torch.nn.Linear(hidden_dim, hidden_dim)
        
        self.gamma = torch.nn.Linear(hidden_dim, 1)
        self.nu = torch.nn.Linear(hidden_dim, 1)
        self.alpha = torch.nn.Linear(hidden_dim, 1)
        self.beta = torch.nn.Linear(hidden_dim, 1)
        self.dropout = torch.nn.Dropout(0.0)

    def forward(self, src):
        fully1 = self.dropout(torch.nn.ReLU()(self.fully1(src)))
        fully2 = self.dropout(torch.nn.ReLU()(self.fully2(fully1)))
        fully3 = self.dropout(torch.nn.ReLU()(self.fully3(fully2)))
        
        gamma = self.gamma(fully3)
        alpha = torch.nn.Softplus()(self.alpha(fully3)) + 1 #+ 1E-9
        beta = torch.nn.Softplus()(self.beta(fully3))
        nu = torch.nn.Softplus()(self.nu(fully3))# + 1E-9
        
        return gamma, nu, alpha, beta


class Network(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=100):
        super(Network, self).__init__()
        self.fully1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fully2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fully3 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.gamma = torch.nn.Linear(hidden_dim, 1)
        self.dropout = torch.nn.Dropout(0.1)

    def forward(self, src):
        self.fully1_val = self.dropout(torch.nn.Tanh()(self.fully1(src)))
        self.fully2_val = self.dropout(torch.nn.Tanh()(self.fully2(self.fully1_val)))
        self.fully3_val = self.dropout(torch.nn.Tanh()(self.fully3(self.fully2_val)))
        
        gamma = self.gamma(self.fully3_val)
        return gamma
    
    def forward_s(self, src, scale=1, bias=0, s=30):
        results = []
        self.train()
        for i in range(s):
            results.append(self.forward(src)*scale + bias)
        results = torch.stack(results)
        self.eval()
        return torch.mean(results, axis=0), torch.std(results, axis=0)

    
class EvidentialnetMarginalLikelihood(torch.nn.modules.loss._Loss):
    """
    Marginal likelihood error of prior network.
    The target value is not a distribution (mu, std), but a just value.
    
    This is a negative log marginal likelihood, with integral mu and sigma.
    
    """
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean', eps=1e-9):
        super(EvidentialnetMarginalLikelihood, self).__init__(size_average, reduce, reduction)
    
    def forward(self, gamma: torch.Tensor, nu: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            gamma, nu, alpha, beta -> outputs of prior network
            
            target -> target value
            
        Return:
            (Tensor) Negative log marginal likelihood of EvidentialNet
                p(y|m) = Student-t(y; gamma, (beta(1+nu))/(nu*alpha) , 2*alpha)

                then, the negative log likelihood is (CAUTION QUITE COMPLEX!)

                NLL = -log(p(y|m)) =
                    log(3.14/nu)*0.5 - alpha*log(2*beta*(1 + nu)) + (alpha + 0.5)*log( nu(target - gamma)^2 + 2*beta(1 + nu) )
                    + log(GammaFunc(alpha)/GammaFunc(alpha + 0.5))
        """
        pi = torch.tensor(3.141592741012573)
        
        x1 = torch.log(pi/nu)*0.5
        x2 = -alpha*torch.log(2.*beta*(1.+ nu))
        x3 = (alpha + 0.5)*torch.log( nu*(target - gamma)**2 + 2.*beta*(1. + nu) )
        x4 = torch.lgamma(alpha) - torch.lgamma(alpha + 0.5)
        
        return x1 + x2 + x3 + x4

    
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
        return torch.abs(target - gamma)*(2*nu + alpha) * self.factor
        #return (2*nu + alpha) * self.factor
    
class GaussianNLL(torch.nn.modules.loss._Loss):
    """
    Negative Gaussian likelihood loss.
    """
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean'):
        super(GaussianNLL, self).__init__(size_average, reduce, reduction)
    
    def forward(self, input_mu: torch.Tensor, input_std: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        x1 = 0.5*torch.log(2*np.pi*input_std*input_std)
        x2 = 0.5/(input_std**2)*((target - input_mu)**2)
        
        if self.reduction == 'mean':
            return torch.mean(x1 + x2)
        elif self.reduction == 'sum':
            return torch.sum(x1 + x2)