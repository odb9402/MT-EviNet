"""
This source includs two types of classes or functions.

1. Evaluation methods:
    This source includs some evaluation metrics, such as r-square, CI, mse ...
    or even confidence intervals and ECE, which are uncertainty measures.
    
2. Some plotting functions.

3. Multi-task learning utility functions.
    These functions usually modify gradients of a model itself.
    This adjusting gradient is commonly essential task to produce multi-task learning.

"""
from lifelines.utils import concordance_index
from scipy import stats
from scipy.stats import t, norm, gamma
from scipy.special import logsumexp
from abc import ABC, abstractmethod
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')

device = 'cuda'

to_np = lambda tensor: tensor.cpu().detach().numpy()

def clean_up_grad(tensors):
    for t in tensors:
        t.grad.data.zero_()


def get_mse_coef_test(gamma, nu, alpha, beta, y, tau=1.):
    alpha_eff = check_mse_efficiency_alpha(gamma, nu, alpha, beta, y)
    nu_eff = check_mse_efficiency_nu(gamma, nu, alpha, beta, y)
    delta = (gamma - y).abs()
    min_bound = torch.min(nu_eff, alpha_eff).min()
    c = (min_bound.sqrt()/delta).detach()
    return torch.clip(c, min=False, max=1.)


def get_multi_obj_coef(model, objective_a, objective_b, a_args, b_args, **kwargs):
    """
    Get multi-objective optimization coefficient for two loss functions.
    We can use a combined two loss functions with the returned coefficient
    as like:
    
        LOSS = c * f_a + (1 - c) * f_b
        
    Where 'c' is the return value of this function.
    
    Args:
        model(torch.Module) A model to get a gradient.
        objective_a(torch.nn.modules.loss._Loss) A first loss function.
        objective_b(torch.nn.modules.loss._Loss) A second loss function.
        a_args(Sequence[torch.Tensor]) A loss-function arguments(tensor ouputs of the model)
                                        of the objective_a.
        b_args(Sequence[torch.Tensor]) A loss-function arguments(tensor ouputs of the model)
                                        of the objective_b.
         
    Return:
        (torch.Tensor) A multi-objective coefficient of two functions.
        
    """
    objective_a(*a_args).mean().backward(retain_graph=True)
    grad_a = get_gradient_vector(model, **kwargs)
    try:
        clean_up_grad([*a_args])
    except AttributeError:
        pass
    model.zero_grad()
    
    objective_b(*b_args).mean().backward(retain_graph=True)
    grad_b = get_gradient_vector(model, **kwargs)
    try:
        clean_up_grad([*b_args])
    except AttributeError: ## Attribute error for the output tensor (No grad)
        pass
    model.zero_grad()
    
    return get_mtl_min_norm(grad_a, grad_b), grad_a, grad_b


def check_mse_efficiency_alpha(gamma, nu, alpha, beta, y, reduction='mean'):
    """
    Check the MSE loss (gamma - y)^2 can make negative gradients for alpha, which is
    a pseudo observation of the normal-inverse-gamma. We can use this to check the MSE
    loss can success(increase the pseudo observation, alpha).
    
    Args:
        gamma, nu, alpha, beta(torch.Tensor) output values of the evidential network
        y(torch.Tensor) the ground truth
    
    Return:
        partial f / partial alpha(numpy.array) 
        where f => the NLL loss (BayesianDTI.loss.MarginalLikelihood)
    
    """
    delta = (y-gamma)**2
    right = (torch.exp((torch.digamma(alpha+0.5)-torch.digamma(alpha))) - 1)*2*beta*(1+nu) / nu

    return (right).detach()


def check_mse_efficiency_nu(gamma, nu, alpha, beta, y):
    """
    Check the MSE loss (gamma - y)^2 can make negative gradients for nu, which is
    a pseudo observation of the normal-inverse-gamma. We can use this to check the MSE
    loss can success(increase the pseudo observation, nu).
    
    Args:
        gamma, nu, alpha, beta(torch.Tensor) output values of the evidential network
        y(torch.Tensor) the ground truth
    
    Return:
        partial f / partial nu(torch.Tensor) 
        where f => the NLL loss (BayesianDTI.loss.MarginalLikelihood)
    """
    gamma, nu, alpha, beta = gamma.detach(), nu.detach(), alpha.detach(), beta.detach()
    nu_1 = (nu+1)/nu
    return (beta*nu_1/alpha)


def share_grad_to_zero(model, pass_param_name=('gamma', 'nu', 'alpha','beta')):
    """
    Make gradient vectors of modules in the model be zeros.
    However, if the name of parameter is in 'pass_param_name', the gradient is not
    going to be zeros.
    
    Args:
        model(torch.nn.Module) torch module to be gradient-masking.
        pass_param_name(Sequence(str)) the names of parameters to avoid being zeros.
        
    Return:
        None. The given 'model' will be modified.
    """
    for name, module in model.named_modules():
        if name == '': # Model itself.
            continue
            
        num_children = len(list(module.children()))
        if num_children == 0:
            if len(set.intersection(set(name.split('.')),set(pass_param_name))) > 0:
                pass
            else:
                module.zero_grad()


def get_grad_dropout_prob(grad_tensors):
    """
    Get gradient dropout probability with respect to the sequence tensors.
    
    Args:
        grad_tensors(Sequence(torch.Tensor)) Tensors includes gradients.
        
    Return:
        torch.Tensor, which has the same tensor dimension with input tensors
    """
    grad_tensors = torch.stack(grad_tensors)
    sum_tensors = torch.sum(grad_tensors, axis=0)
    abs_sum_tensors = torch.sum(torch.abs(grad_tensors), axis=0)
    
    grad_dropout_prob = (1 + sum_tensors/abs_sum_tensors)/2
    
    if grad_dropout_prob.shape != grad_tensors[0].shape:
        raise "The final dropout probability shape is different with input tensor shapes"
    else:
        return grad_dropout_prob
    
    
def grad_dropout(grad_tensors):
    """
    Perform Gradient Sign Dropout algorithm of NIPS2020: 
    "Just Pick a Sign: Optimizing Deep Multitask Models with Gradient Sign Dropout"
    
    Args:
        grad_tensors(Sequence(torch.Tensor)) Tensors includes gradients.
    
    Return:
        torch.Tensor, which has the same tensor dimension with input tensors.
        The return tensor represents a new gradient.
    """
    new_grad = torch.zeros(grad_tensors[0].shape, device=grad_tensors[0].device)
    P = get_grad_dropout_prob(grad_tensors)
    
    for grad_tensor in grad_tensors:
        grad_pos = torch.nn.ReLU()(grad_tensor)
        grad_neg = -torch.nn.ReLU()(-grad_tensor)
        
        M = (P > torch.rand(P.shape, device=P.device)).float() * grad_tensor
        M = M + (P < torch.rand(P.shape, device=P.device)).float() * grad_tensor
        
        new_grad += M
    
    return new_grad


def get_gradient_vector(model, pass_param_name=('')):
    """
    Get 1 dimensional gradient vector of the given model.
    
    Args:
        model(torch.nn.Module)
        
    Return:
        torch.FloatTensor - the gradients of weights for the given model
    """
    tensors = []
    for n, w in model.named_parameters():
        if not len(set.intersection(set(n.split('.')),set(pass_param_name))) > 0:
            #print(n,w)
            tensors.append(w.grad.flatten())
        
    grad = torch.cat(tensors)
    return grad


def get_mtl_min_norm(a, b, reverse=False):
    """
    For two gradients of two tasks (MSE training, NLL training),
    get_mtl_min_norm() calculates the linear-combination coefficient which makes
    the l2 norm gradients are minimum value.
    
    If a, b are gradient vectors and c its linear coefficient, the return value c_opt is:
    
        c_opt = argmin(c1) ||c*a + (1 - c)*a||_2
    
    Where ||X||_2 is L2 norm of X.
    This optimization problem has the analytical solution, which is implemented in this function.
    
    Args:
        a(torch.Tensor): Gradients of task A.
        b(torch.Tensor): Gradients of task B.
        reverse(bool): If true, the larger gradient will be choiced
            when the gradient a and b are not conflict. 
    Return:
        torch.FloatTensor: a scalar value represents 'c_opt' above.
    """
    coefficient = torch.clamp(((b - a)@b)/torch.sum((a - b)**2), min=0, max=1)
    if not reverse:
        return coefficient
    else:
        return 1 - coefficient 

def gradient_scaling(model, alpha):
    """
    Scale alpha times the gradients of the models.
    
    for ever paramter w,
        dw/dl = dw/dl * alpha
    
    Args:
        model(torch.nn.Module)
        alpha(torch.nn.FloatTensor, Float)
    
    Return:
        None
    """
    for w in model.parameters():
        w.grad *= alpha


class DTItrainer(ABC):
    # TODO, or should I?...
    def train(self, model, dataloader, valid_dataloader, objective_fn, opt,
                epochs=100, **kwargs):
        self.total_loss = 0.
        self.loss_history = []
        self.valid_loss_history = []
        self.model = model
        self.epochs = epochs
        
        self.train_loop(epoch, **kwargs)
            
    @abstractmethod
    def train_loop(self, epoch, **kwargs):
        pass
    
    def preconditions(self):
        pass


def mcdrop_nll(y, preds, sample_num=5, prior_length_scale=0.01, tau=1.):
    """
    Calculate likelihood for the MC-dropout.
    The 'tau' constant will be calculated as the precision(variance),
    which is noted at the supplementray of the paper "Dropout as the Bayesian approximation; ICML 2016; Gal et al."


    Args:
        y (np.array[float]): Numpy array of ground truth values
        preds (np.array[float]): Numpy array of predictions
        decay_rate ([float]): Weight decay rate
        prob ([float]): Dropout probability 
        sample_size ([type]): Total sample size (not a number of batch, but a number of sample)
        sample_num (int, optional): Number of samples of differently predicted values. Defaults to 5.

    Returns:
        [float]: The likelihood. 
    """
    tau = tau#(1-prob)*(prior_length_scale**2)/(2*decay_rate)
    distance = (y - preds)**2 ## (N , T)
    x1 = logsumexp((-0.5*tau*distance), axis=1) ## (N, 1)
    x2 = -np.log(sample_num) -0.5*np.log(2*np.pi) + 0.5*np.log(tau)

    return -np.mean(x1 + x2)


def train_SWAG_model(model, dataloader, valid_dataloader, objective_fn, opt,
                     epochs=100, swag_start=80, max_models=20, verbose=True):
    """
    Training SWAG model with the base model "model" argument. Note that the given base model
    does not have to be trained since this process includes.
    
    Args:
        model(torch.nn.Module): 
            base model for SWAG training
        
        dataloader(torch.dataloader):
            training dataloader
        
        valid_dataloader(torch.dataloader):
            validation dataloader
            
        objective_fn(torch.nn.modules.loss._Loss):
            loss function
        
        opt(torch.optim.optimizer):
            optimizer for training
        
        epochs(float):
        
        swag_start(int):
            Epoch to start SWAG training. If the training epoch is under the 'swag_start',
            the SWAG model does not collect the model weights(does not train).
        
        max_models(int):
            Number of weight histories for SWAG covariance matrix. If max_models=20, 
            weight histories only from the last 20 epochs will be used for SWAG training
        
        verbose(bool):
            print training information or not
            
    Return:
        swag_model(swag.SWAG):
            trained swag model, which has captured the weight histories of the base model.
            
        histories(dict{List[float]}):
            Loss histories.
    """
    
    histories = dict()
    learning_rate = opt.default.lr
    total_loss = 0.
    i = 1
    loss_history = []
    valid_loss_history = []
    params = model.parameters()
    swag_model = SWAG(model, no_cov_mat=False, max_num_models=max_models).to(device)
    
    for epoch in range(1, epochs):
        lr = schedule(epoch, learning_rate, swag_start)
        print("learning rate : {:.5f}".format(lr))
        swag.utils.adjust_learning_rate(opt, lr)
        model.train()

        ############## Normal SGD Training ####################
        for d, p, y in dataloader:
            opt.zero_grad()
            prediction = model(d.to(device),p.to(device))#interaction_model(p_emb, d_emb)

            loss = objective_fn(y.to(device), prediction.view(-1))
            loss.backward()
            loss_history.append(loss.item())
            opt.step()

            total_loss += loss.item()

            i += 1

        cur_loss = total_loss / len(dataloader)
        total_loss = 0.

        total_valid_loss = 0.
        model.eval()
        for d_v, p_v, y_v in valid_dataloader:
            prediction_v = model(d_v.to(device),p_v.to(device))#interaction_model(p_emb, d_emb)
            loss_v = objective_fn(y_v.to(device), prediction_v.view(-1))
            total_valid_loss += loss_v.item()

        valid_loss = total_valid_loss/len(valid_dataloader)
        valid_loss_history.append(total_valid_loss/len(valid_dataloader))

        ################ SWAG Training ####################
        swag_loss = None
        if epoch + 1 >= swag_start:
            model.eval()
            predictions_v = []
            targets_v = []
            for d_v, p_v, y_v in valid_dataloader:
                prediction_v = model(d_v.to(device),p_v.to(device))#interaction_model(p_emb, d_emb)
                predictions_v.append(prediction_v.view(-1).detach().cpu().numpy())
                targets_v.append(y_v.numpy())
            sgd_preds = np.concatenate(predictions_v)
            sgd_targets = np.concatenate(targets_v)

            sgd_ens_preds = sgd_preds.copy()
            #n_ensembled += 1
            swag_model.collect_model(model)

            loss_history_swag = []
            total_swag_loss = 0.0
            if (epoch + 1) % 5 == 0: ## Evaluate SWAG Model
                print("eval swag model")
                swag_model.sample(0.0) ## Sampling with 0 scale is equivalent with deterministic SWA.
                swag.utils.bn_update(valid_dataloader, swag_model)
                for d_v, p_v, y_v in valid_dataloader:
                    prediction_swag = swag_model(d_v.to(device),p_v.to(device))#interaction_model(p_emb, d_emb)
                    loss_swag = objective_fn(y_v.to(device), prediction_swag.view(-1))
                    total_swag_loss += loss_swag.item()
                    loss_history_swag.append(loss_swag.item())
                swag_loss = total_swag_loss/len(valid_dataloader)
            else:
                swag_loss = None
        
        if verbose:
            if swag_loss != None:
                print("Epoch {}: Train loss [{:.5f}] Val loss [{:.5f}]; SWA loss [{:.5f}]".format(
                    epoch+1, cur_loss, valid_loss, swag_loss))
            else:
                print("Epoch {}: Train loss [{:.5f}] Val loss [{:.5f}]".format(
                    epoch+1, cur_loss, valid_loss))
    
    histories['train'] = loss_history
    histories['valid'] = valid_loss_history
    histories['swag'] = loss_history_swag
    
    return swag_model, histories


def eval_uncertainty(mu, std, Y, confidences=None, sample_num=10, verbose=True, **kwargs):
    """
    "Accurate Uncertainties for Deep Learning Using Clibrated Regression"
    ICML 2020; Kuleshov et al. suggested the ECE-like uncertainty evaluation method
    for uncertainty estimation of regression-tasks.
    Note that the calibration metric(error) is in 3.5, Calibration section of the paper.
    
    p_j = <0.1, 0.2, ..., 0.9>
    Acc(p_j) = The accuracy of true values being inside between confidence interval "p_j"
    
    Args:
        mu(np.array): Numpy array of the predictive mean
        std(np.array): Numpy array of the predictive standard derivation
        Y(np.arry): Numpy array of ground truths
        confidences
        sample_num(Int or np.array): Number of samples to calculate t-distributions. If None, it uses Normal.
    
    Return: 
        Tuple(Metric = sum[ (p_j - ACC(p_j))^2 ], List of confidence errors for each confidence level)
    """
    if confidences == None:
        confidences = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    calibration_errors = []
    interval_acc_list = []
    for confidence in confidences:
        low_interval, up_interval = confidence_interval(mu, std, sample_num, confidence=confidence)
        hit = 0
        for i in range(len(Y)):
            if low_interval[i] <= Y[i] and Y[i] <= up_interval[i]:
                hit += 1
        
        interval_acc = hit/len(Y)
        interval_acc_list.append(interval_acc)
        
        if verbose:
            print("Interval acc: {}, confidence level: {}".format(interval_acc, confidence))
        calibration_errors.append((confidence - interval_acc)**2)
    
    return sum(calibration_errors), calibration_errors, interval_acc_list
    

def confidence_interval(mu, std, sample_num, confidence=0.9):
    """
    Calculate confidence interval from mean and std for each predictions
    under the empricial t-distribution.
    
    If the sample_num is given as the 0, it will compute Gaussian confidence interval,
    not t-distribution
    
    Args:
        mu(np.array): Numpy array of the predictive mean
        std(np.array): Numpy array of the predictive standard derivation
    
    Return:
        low_interval(np.array), up_interval(np.array): confidence intervals
    """
    n = sample_num
    if type(sample_num) == np.ndarray:
        h = std * t.ppf((1 + confidence) / 2, n - 1)
    else:
        if sample_num != None:
            h = std * t.ppf((1 + confidence) / 2, n - 1)
        else:
            h = std * norm.ppf((1 + confidence) / 2)
    low_interval = mu - h
    up_interval = mu + h
    return low_interval, up_interval


def schedule(epoch, lr_init, swag_start, swa_lr_factor=2):
    """
    SWA Gaussian learning rate schedule.
    """
    swa_lr = lr_init/swa_lr_factor
    t = (epoch) / (swag_start)
    lr_ratio = swa_lr / lr_init
    if t <= 0.5:
        factor = 1.0
    elif t <= 0.9:
        factor = 1.0 - (1.0 - lr_ratio) * (t - 0.5) / 0.4
    else:
        factor = lr_ratio
    return lr_init * factor


def plot_predictions(y, preds, std=[], title="Predictions", savefig=None,
                     interval_level=0.9, v_max=None, v_min=None, sample_num=30,
                     rep_conf="bar", post_setting=None, **kwargs):
    """
    Show a scatter plot for prediction accuracy and if given, confidence levels.
    
    Args:
        y(np.array) ground truth predictions.
        
        preds(np.array) predicted values.
        
        std(np.array) predicted std, if given, draw confidence interval.
        
        title(str) The title of the plot.
        
        savefig(str) The name of output figure file. If not given, just show the figure.
        
        interval_level(float) The level of confidence interval, e. g) 0.9 will show 90%
            confidence interval
        
        v_max, v_min(float, float) minimum, maximum value of x, y axises.
        
        sample_num(float or np.array) Sampling number to calculate confidence level using
            t-distribution. It is a freedom parameter of t-distribution. You can assign
            different freedoms(sample_num) for each sample by using np.array as the sample_num.
            
        rep_conf('bar' or 'color') If 'bar' choosen, the confidence interval will be represented
            as bars. If 'color' choosen, it will be colors for dots.
        
        post_setting(func) This function 
        
        **kwargs -> You can pass some arguments for matplotlib plot functions.
        
    Return:
        None
    """
    if v_min == None:
        v_min = min(min(y), min(preds))*0.98
    if v_max == None:
        v_max = max(max(y), max(preds))*1.02
    
    fig = plt.figure(figsize=(10,10))
    if len(std) == 0:
        plt.scatter(y, preds, **kwargs)
    else:
        if rep_conf == 'bar':
            plt.errorbar(y, preds,
                     yerr=(confidence_interval(preds, std, sample_num, interval_level)-preds)[1],
                     fmt='o',
                     ecolor='black',
                     elinewidth=1,
                     **kwargs)
        elif rep_conf == 'color':
            fig = plt.figure(figsize=(12,10))
            plt.scatter(y, preds, c=std, **kwargs)
            plt.colorbar()
        else:
            print("rep_conf should be either 'bar' or 'color'. Not {}".format(rep_conf))
            return None
    
    plt.axline((v_min, v_min), (v_max,v_max), color='black', linestyle='--')
    plt.xlabel("Ground truth")
    plt.ylabel("Prediction")
    plt.ylim(bottom=v_min, top=v_max)
    plt.title(title, fontsize=20)
    if post_setting != None:
        post_setting()
    
    if savefig != None:
        plt.savefig(savefig)
    else:
        plt.show()
        

def aupr(Y, P, data_type='kiba', **kwargs):
    from sklearn import metrics
    if data_type=='kiba':
        threshold = 12.1
    elif data_type=='davis':
        threshold = 7
    Y = np.copy(Y)
    Y[Y < threshold] = 0
    Y[Y >= threshold] = 1
    
    return metrics.average_precision_score(Y, P)
        
"""
Metrics from DeepDTA sources
"""
def get_cindex(Y, P):
    """
    ******NOTE******
    
    Now the get_cindex is invalid (order dependent of given pairs).
    We will use lifelines.utils for 
    """
    summ = 0
    pair = 0
    
    for i in range(0, len(Y)):
        for j in range(0, len(Y)):
            if i != j:
                if(Y[i] > Y[j]):
                    pair +=1
                    summ +=  1* (P[i] > P[j]) + 0.5 * (P[i] == P[j])
            
    if pair != 0:
        return summ/pair
    else:
        return 0


def r_squared_error(y_obs,y_pred):
    """
    Calculate r^2, r-square metric, which is commonly used for QSAR evaluation.
    
    
    """
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    y_pred_mean = [np.mean(y_pred) for y in y_pred]

    mult = sum((y_pred - y_pred_mean) * (y_obs - y_obs_mean))
    mult = mult * mult

    y_obs_sq = sum((y_obs - y_obs_mean)*(y_obs - y_obs_mean))
    y_pred_sq = sum((y_pred - y_pred_mean) * (y_pred - y_pred_mean) )

    return mult / float(y_obs_sq * y_pred_sq)

##################################################################

def log_likelihood(Y, preds, scale, sample_num=30, **kwargs):
    """
    Calculate negative log likelihood
    """
    
    if type(sample_num) == np.ndarray: 
        nll_t = lambda y, mu, std, freedom: np.log(t.pdf(y, freedom, mu, std))
    else:
        if sample_num == None:
            nll_t = lambda y, mu, std: np.log(norm.ppf(y, mu, std))
        else:
            nll_t = lambda y, mu, std: np.log(t.pdf(y, sample_num, mu, std))
        
    nll_values = []
    for i in range(len(Y)):
        if type(sample_num) == np.ndarray: 
            nll_values.append(nll_t(Y[i], preds[i], scale[i], sample_num[i]))
        else:
            nll_values.append(nll_t(Y[i], preds[i], scale[i]))
    return np.mean(nll_values)


def evaluate_model(Y, preds, std=[], **kwargs):
    """
    Evaluate model for various metrics.
    All keyword arguments will be passed for uncertainty evaluations.
    
    Args:
        Y(np.array) ground truth affinity values
        preds(np.array) predicted affinity values
        std(np.array or List) predicted standard derivation
        
    Return:
        result(dict) Dictionary including the evaluation results.
            keys: 'MSE', 'r2', 'CI', 'cal_error', 'cal_errors'
    """
    results = dict()
    
    results['MSE'] = str(np.mean((Y - preds)**2))
    results['r2'] = str(r_squared_error(Y, preds))
    results['CI'] = str(concordance_index(Y, preds))
    if len(std) != 0:
        results['cal_error'], cal_errors, _ = eval_uncertainty(preds, std, Y, **kwargs)
        results['LL'] = log_likelihood(Y, preds, std, **kwargs)
        results['ECE'] = str(np.mean(np.sqrt(cal_errors)))
        results['cal_error'] = str(results['cal_error'])
    results['AUPR'] = str(aupr(Y, preds, **kwargs))
    
    return results


def calibaration_eval(Y, preds, std):
    eval_uncertainty(preds, std, Y)
##########################################################################
import string
from itertools import cycle
from six.moves import zip

def label_axes(fig, labels=None, loc=None, **kwargs):
    """
    Walks through axes and labels each.

    kwargs are collected and passed to `annotate`

    Parameters
    ----------
    fig : Figure
         Figure object to work on

    labels : iterable or None
        iterable of strings to use to label the axes.
        If None, lower case letters are used.

    loc : len=2 tuple of floats
        Where to put the label in axes-fraction units
    """
    if labels is None:
        labels = string.ascii_lowercase

    # re-use labels rather than stop labeling
    labels = cycle(labels)
    if loc is None:
        loc = (.9, .9)
    for ax, lab in zip(fig.axes, labels):
        ax.annotate(lab, xy=loc,
                    xycoords='axes fraction',
                    **kwargs)
        
        
def fit_gamma_std(std):
    """
    Fit gamma function for given standard deviation.
    
    Args:
        std(np.array): A numpy array contains the standard deviation.
        
    Return:
        alpha, loc, beta
    """
    fit_alpha, fit_loc, fit_beta = stats.gamma.fit(std)
    return fit_alpha, fit_loc, fit_beta


def get_p_values(std, alpha, loc, beta, tensor=True):
    """
    Get p-values for std, under the gamma distribution parameterized with
    (alpha, loc, beta)

    Args:
        std(np.array): A numpy array contains the standard deviation.

    Return:
        np.array: A numpy array includes p-values for given data

    """
    return torch.Tensor(gamma.sf(std, alpha, loc, beta))
