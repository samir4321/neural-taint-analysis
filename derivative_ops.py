# derivative_ops.py
import torch
import numpy as np

def sensitivity(f, x):
    """
    frobenius norm of jacobian matrix for each passed x (sensitivity)
    :param f:
    :param x:
    :return:
    """
    jac = jacobian(f, x)
    return torch.sqrt(torch.sum(torch.sum(jac**2, dim=1), dim=1))


def mean_sensitivity(f, x):
    return torch.mean(sensitivity(f, x), axis=0)


def saliency_map(f, x):
    return torch.abs(jacobian(f, x))


def top_mean_saliencies(f, x, top_k):
    """
    top (i, j, S_ij averaged across x)
    """
    a = mean_saliency_map(f, x).data.numpy()
    top_indices = tuple(
        zip(*np.unravel_index(a.argsort(axis=None), dims=a.shape)))[::-1][
                  :top_k]
    return [tuple(list(ix) + [a[ix]]) for ix in top_indices]

def top_saliencies(f, x, top_k):
    """
        Given S_ij(x) saliency map, returns
        (x_index, i, j, S_ij(x)) for  top_k S_ij(x) values
    """
    a = saliency_map(f, x).data.numpy()
    #TODO("argsort very slow for large arrays")
    top_indices = tuple(zip(*np.unravel_index(a.argsort(axis=None), dims=a.shape)))[::-1][:top_k]
    return [tuple(list(ix) + [a[ix]]) for ix in top_indices]

def mean_saliency_map(f, x):
    return torch.mean(saliency_map(f, x), axis=0)

def jacobian(f, x):
    # !! this is actually the same as the jacobian !!
    N = x.shape[0]
    x.requires_grad_(True)
    y = f(x)
    D_in = x.shape[1]
    D_out = y.shape[1]
    jac = torch.zeros(N, D_out, D_in)
    for i in range(D_out):
        jac[:, i, :] = torch.autograd.grad(y[:, i].sum(), x, retain_graph=True)[
            0] # .data
    return jac


def hessian(f, x):
    # TODO("fix hessian")
    # returns one hessian for each sample passed in x
    N = x.shape[0]
    x.requires_grad_(True)
    y = f(x)
    D_in = x.shape[1]
    D_out = y.shape[1]
    jac = torch.zeros(size=(N, D_in, D_out))
    hess = torch.zeros(size=(N, D_in, D_in, D_out))
    for n in range(N):
        xv = x[n, :]
        xv.requires_grad_(True)
        yv = f(xv)
        for i in range(D_out):
            jac[n, i, :] = torch.autograd.grad(yv[i], xv, create_graph=True,
                                               retain_graph=True)[
                0]  # .data
            for i in range(D_in):
                jac_out = jac[n, i, i]
                jac_out.requires_grad_(True)
                hess[n, i, :, i] = torch.autograd.grad(jac_out, xv, create_graph=True,
                                               retain_graph=True)[0]
    return hess
